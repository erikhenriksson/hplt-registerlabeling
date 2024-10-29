import os

os.environ["HF_HOME"] = ".hf/hf_home"
import json
import time
import zstandard as zstd
import io
import torch
from tqdm.auto import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import sigmoid
from itertools import islice
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# Labels structure
labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]
child_to_parent = {
    child: parent for parent, children in labels_structure.items() for child in children
}
label_to_index = {label: idx for idx, label in enumerate(labels_all)}


def local_data_adaptive(file_path, rank, world_size):
    """Process data while preserving original line order"""
    buffer = []
    BUFFER_SIZE = 1000
    line_count = 0

    with open(file_path, "rb") as file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(file) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text_reader:
                if line_count % world_size == rank:
                    # Store original position with the line
                    buffer.append((line_count, line))
                    if len(buffer) >= BUFFER_SIZE:
                        yield from buffer
                        buffer = []
                line_count += 1

            if buffer:
                yield from buffer


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def process_labels(
    item_data, prob, pred_label, labels_all, child_to_parent, label_to_index
):
    """Optimized label processing using sets"""
    labels_indexes = pred_label.tolist()

    active_labels = set()
    for i, is_active in enumerate(labels_indexes):
        if is_active == 1:
            label = labels_all[i]
            active_labels.add(label)
            if label in child_to_parent:
                active_labels.add(child_to_parent[label])

    item_data["registers"] = list(active_labels)
    item_data["register_probabilities"] = [round(float(p), 4) for p in prob.tolist()]
    return item_data


def batch_process(
    rank,
    world_size,
    model,
    tokenizer,
    input_path,
    batch_size=128,
    max_batch_length=12800,
):
    data_iterator = local_data_adaptive(input_path, rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    max_length = 512
    input_ids_buffer = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    attention_mask_buffer = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )

    processed_count = 0

    while True:
        large_batch = list(islice(data_iterator, max_batch_length))
        if not large_batch:
            break

        # Unpack the position and line data
        positions, lines = zip(*large_batch)
        text_data = [(idx, json.loads(line)) for idx, line in zip(positions, lines)]
        texts = [item[1]["text"] for item in text_data]

        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        lengths = [len(ids) for ids in encodings["input_ids"]]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        processed_results = []

        for i in range(0, len(sorted_indices), batch_size):
            current_batch_size = min(batch_size, len(sorted_indices) - i)
            batch_indices = sorted_indices[i : i + current_batch_size]

            batch_items = [text_data[idx][1] for idx in batch_indices]
            batch_positions = [text_data[idx][0] for idx in batch_indices]
            batch_encodings = {
                key: [encodings[key][idx] for idx in batch_indices]
                for key in encodings.keys()
            }

            # Get the maximum length in this batch
            current_length = max(len(seq) for seq in batch_encodings["input_ids"])

            # Reset buffers to zero
            input_ids_buffer.zero_()
            attention_mask_buffer.zero_()

            # Fill the pre-allocated buffers
            for j, seq in enumerate(batch_encodings["input_ids"]):
                seq_len = len(seq)
                input_ids_buffer[j, :seq_len].copy_(
                    torch.tensor(seq, dtype=torch.long, device=device)
                )
                attention_mask_buffer[j, :seq_len] = 1

            # Use views of the buffers for the current batch
            input_ids = input_ids_buffer[:current_batch_size, :current_length]
            attention_mask = attention_mask_buffer[:current_batch_size, :current_length]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = sigmoid(outputs.logits)
                predicted_labels = (probabilities > 0.5).int()

            for idx, (item_data, prob, pred_label, orig_pos) in enumerate(
                zip(batch_items, probabilities, predicted_labels, batch_positions)
            ):
                processed_item = process_labels(
                    item_data.copy(),
                    prob,
                    pred_label,
                    labels_all,
                    child_to_parent,
                    label_to_index,
                )
                processed_results.append((orig_pos, processed_item))
                processed_count += 1

            if rank == 0 and processed_count % 1000 == 0:
                print(f"GPU {rank}: Processed {processed_count} items")

        # Sort by original position before yielding
        processed_results.sort(key=lambda x: x[0])
        yield processed_results

    print(f"GPU {rank}: Completed processing {processed_count} items")


class OrderedFileReader:
    """Helper class to read and track items from a file with their positions"""

    def __init__(self, file_path):
        self.file = open(file_path, "rb")
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.file)
        self.text_reader = io.TextIOWrapper(self.reader, encoding="utf-8")
        self.current_item = None
        self.exhausted = False
        self._read_next()

    def _read_next(self):
        try:
            line = self.text_reader.readline()
            if not line:
                self.exhausted = True
                self.current_item = None
            else:
                item = json.loads(line)
                self.current_item = (item["original_position"], item["data"])
        except Exception as e:
            print(f"Error reading line: {e}")
            self.exhausted = True
            self.current_item = None

    def get_current(self):
        return self.current_item

    def advance(self):
        self._read_next()

    def close(self):
        self.text_reader.close()
        self.reader.close()
        self.file.close()


def merge_ordered_files(temp_files, output_path, buffer_size=8 * 1024 * 1024):
    """Merge multiple ordered files while maintaining global order"""
    # Initialize readers for all temp files
    readers = [OrderedFileReader(f) for f in temp_files]
    active_readers = len(readers)

    with open(output_path, "wb", buffering=buffer_size) as final_out:
        cctx = zstd.ZstdCompressor(level=1)
        with cctx.stream_writer(final_out) as final_writer:
            output_position = 0
            string_buffer = []
            string_buffer_size = 0

            while active_readers > 0:
                # Find reader with smallest current position
                min_pos = float("inf")
                min_reader_idx = -1

                for idx, reader in enumerate(readers):
                    if not reader.exhausted:
                        current = reader.get_current()
                        if current and current[0] < min_pos:
                            min_pos = current[0]
                            min_reader_idx = idx

                if min_reader_idx == -1:
                    break

                # Verify we're writing in correct order
                assert (
                    min_pos == output_position
                ), f"Position mismatch: expected {output_position}, got {min_pos}"

                # Write the item
                pos, item = readers[min_reader_idx].get_current()
                line = json.dumps(item) + "\n"
                string_buffer.append(line)
                string_buffer_size += len(line)

                if string_buffer_size >= buffer_size:
                    final_writer.write("".join(string_buffer).encode("utf-8"))
                    string_buffer = []
                    string_buffer_size = 0

                # Advance the reader
                readers[min_reader_idx].advance()
                if readers[min_reader_idx].exhausted:
                    active_readers -= 1

                output_position += 1

            # Write any remaining items
            if string_buffer:
                final_writer.write("".join(string_buffer).encode("utf-8"))

    # Close and cleanup readers
    for reader in readers:
        reader.close()


def get_zstd_compression_level(file_path):
    """Detect the compression level of a zstd file"""
    with open(file_path, "rb") as f:
        # Read frame header
        header = f.read(4)
        if header.startswith(b"\x28\xB5\x2F\xFD"):  # zstd magic number
            # Read frame header descriptor
            frame_header = f.read(1)[0]
            # Extract compression level
            level = (frame_header >> 3) & 0x1F
            return level
        else:
            print("Warning: Could not detect compression level, using default")
            return None


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    temp_output_path = f"{cfg.output_path}.temp_{rank}"

    start_time = time.time()
    if rank == 0:
        print(f"Starting processing on {world_size} GPUs")

    # Optimized writing with large buffers
    WRITE_BUFFER_SIZE = 8 * 1024 * 1024  # 8MB buffer
    string_buffer = []
    string_buffer_size = 0
    MAX_BUFFER_SIZE = 16 * 1024 * 1024  # 16MB

    with open(temp_output_path, "wb", buffering=WRITE_BUFFER_SIZE) as out_file:
        cctx = zstd.ZstdCompressor(level=1)
        with cctx.stream_writer(out_file) as writer:
            for processed_batch in batch_process(
                rank,
                world_size,
                model,
                tokenizer,
                cfg.input_path,
                cfg.batch_size,
                cfg.max_batch_length,
            ):
                # Sort batch by original position before writing
                processed_batch.sort(key=lambda x: x[0])

                for orig_pos, item in processed_batch:
                    output_item = {"original_position": orig_pos, "data": item}
                    json_str = json.dumps(output_item) + "\n"
                    string_buffer.append(json_str)
                    string_buffer_size += len(json_str)

                    if string_buffer_size >= MAX_BUFFER_SIZE:
                        writer.write("".join(string_buffer).encode("utf-8"))
                        string_buffer = []
                        string_buffer_size = 0

            if string_buffer:
                writer.write("".join(string_buffer).encode("utf-8"))

    end_time = time.time()
    print(f"GPU {rank}: Finished processing in {end_time - start_time:.2f} seconds")

    dist.barrier()

    if rank == 0:
        print("Combining results from all GPUs...")
        temp_files = [f"{cfg.output_path}.temp_{r}" for r in range(world_size)]
        merge_ordered_files(temp_files, cfg.output_path)
        print("Results combined successfully!")

        # Clean up temp files
        for temp_file in temp_files:
            os.remove(temp_file)

    cleanup()


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batch_length", type=int, default=1000)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/en/1_sample.jsonl.zst")
    parser.add_argument(
        "--output_path", default="data/en/1_sample_register_labels.jsonl.zst"
    )
    cfg = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")
    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

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


def count_lines_in_zst(file_path):
    count = 0
    with open(file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for _ in text_stream:
                count += 1
    return count


def local_data_adaptive(file_path, rank, world_size, chunk_size_mb=1024):
    """Process data in chunks without knowing total line count"""
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // world_size
    start_pos = rank * chunk_size
    end_pos = start_pos + chunk_size if rank != world_size - 1 else file_size

    with open(file_path, "rb") as file:
        file.seek(start_pos)

        # If not first chunk, read until next newline
        if rank > 0:
            file.readline()  # Skip partial line

        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(file) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")

            while file.tell() < end_pos:
                line = text_reader.readline()
                if not line:
                    break
                yield line


def local_data(file_path, rank, world_size, total_lines):
    """Optimized data loading with buffering"""
    chunk_size = total_lines // world_size
    start_line = rank * chunk_size
    end_line = start_line + chunk_size if rank != world_size - 1 else total_lines

    buffer = []
    BUFFER_SIZE = 1000

    current_line = 0
    with open(file_path, "rb") as file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(file) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                if start_line <= current_line < end_line:
                    buffer.append(line)
                    if len(buffer) >= BUFFER_SIZE:
                        yield from buffer
                        buffer = []
                elif current_line >= end_line:
                    break
                current_line += 1

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
    # total_lines,
    batch_size=128,
    max_batch_length=12800,
):
    data_iterator = local_data_adaptive(input_path, rank, world_size, total_lines)
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    # Set up progress bar
    pbar = None
    if rank == 0:
        pbar = tqdm(total=total_lines // world_size, desc=f"GPU {rank}")

    # Pre-allocate GPU memory for batch tensors
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

        text_data = [(idx, json.loads(line)) for idx, line in enumerate(large_batch)]
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

            for orig_idx, (item_data, prob, pred_label) in enumerate(
                zip(batch_items, probabilities, predicted_labels)
            ):
                processed_item = process_labels(
                    item_data.copy(),
                    prob,
                    pred_label,
                    labels_all,
                    child_to_parent,
                    label_to_index,
                )
                processed_results.append((batch_indices[orig_idx], processed_item))
                processed_count += 1

            if rank == 0 and pbar is not None:
                pbar.update(current_batch_size)

        processed_results.sort(key=lambda x: x[0])
        yield [item for _, item in processed_results]

    if rank == 0 and pbar is not None:
        pbar.close()


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    """
    if rank == 0:
        total_lines = count_lines_in_zst(cfg.input_path)
        print(f"Total lines to process: {total_lines}")
    else:
        total_lines = None

    total_lines = torch.tensor(total_lines if total_lines is not None else 0).to(device)
    dist.broadcast(total_lines, src=0)
    total_lines = total_lines.item()
    """
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
        cctx = zstd.ZstdCompressor(level=1)  # Faster compression
        with cctx.stream_writer(out_file) as writer:
            for processed_batch in batch_process(
                rank,
                world_size,
                model,
                tokenizer,
                cfg.input_path,
                # total_lines,
                cfg.batch_size,
                cfg.max_batch_length,
            ):
                for item in processed_batch:
                    json_str = json.dumps(item) + "\n"
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
        with open(cfg.output_path, "wb", buffering=WRITE_BUFFER_SIZE) as final_out:
            cctx = zstd.ZstdCompressor(level=1)
            with cctx.stream_writer(final_out) as final_writer:
                for r in range(world_size):
                    temp_file = f"{cfg.output_path}.temp_{r}"
                    with open(temp_file, "rb") as temp_in:
                        dctx = zstd.ZstdDecompressor()
                        with dctx.stream_reader(temp_in) as reader:
                            while True:
                                chunk = reader.read(WRITE_BUFFER_SIZE)
                                if not chunk:
                                    break
                                final_writer.write(chunk)
                    os.remove(temp_file)
        print("Results combined successfully!")

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

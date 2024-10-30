import os
import json
import time
import zstandard as zstd
import io
import torch
from tqdm.auto import tqdm
from argparse import ArgumentParser
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.nn.functional import sigmoid
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# Enable TF32
torch.set_float32_matmul_precision("high")

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


class PreprocessedDataset(Dataset):
    def __init__(self, encodings, original_items, positions):
        self.encodings = encodings
        self.original_items = original_items
        self.positions = positions

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "original_item": self.original_items[idx],
            "position": self.positions[idx],
        }

    def __len__(self):
        return len(self.positions)


def preprocess_file(input_path, output_dir, rank, world_size):
    """Preprocess the zst file and save tokenized data for each GPU"""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    texts = []
    original_items = []
    positions = []

    print(f"GPU {rank}: Starting preprocessing...")

    # Read and decompress the file
    with open(input_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for pos, line in enumerate(text_stream):
                if pos % world_size == rank:  # Only process this GPU's share
                    try:
                        item = json.loads(line)
                        texts.append(item["text"])
                        original_items.append(item)
                        positions.append(pos)
                    except json.JSONDecodeError:
                        continue

    print(f"GPU {rank}: Tokenizing {len(texts)} texts...")

    # Tokenize all texts at once
    encodings = tokenizer(
        texts, truncation=True, max_length=512, padding=False, return_tensors=None
    )

    # Save preprocessed data
    torch.save(
        {
            "encodings": encodings,
            "original_items": original_items,
            "positions": positions,
        },
        f"{output_dir}/preprocessed_{rank}.pt",
    )

    print(f"GPU {rank}: Preprocessing complete")
    return len(texts)


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


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Create preprocessing directory
    preprocess_dir = f"{os.path.dirname(cfg.output_path)}/preprocessed"

    # Stage 1: Preprocess data if not already done
    preprocess_file_path = f"{preprocess_dir}/preprocessed_{rank}.pt"
    if not os.path.exists(preprocess_file_path):
        preprocess_file(cfg.input_path, preprocess_dir, rank, world_size)

    # Stage 2: Load preprocessed data and process
    print(f"GPU {rank}: Loading preprocessed data...")
    data = torch.load(preprocess_file_path)
    dataset = PreprocessedDataset(
        data["encodings"], data["original_items"], data["positions"]
    )

    # Initialize data collator for dynamic batching
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Create dataloader with dynamic batching
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=data_collator,
        shuffle=False,  # Keep original order
    )

    # Process batches
    results = []
    start_time = time.time()

    print(f"GPU {rank}: Starting inference...")
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=rank != 0):
            # Move inputs to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = sigmoid(outputs.logits)
            predicted_labels = (probabilities > 0.5).int()

            # Process results
            for idx in range(len(batch["position"])):
                processed_item = process_labels(
                    batch["original_item"][idx].copy(),
                    probabilities[idx],
                    predicted_labels[idx],
                    labels_all,
                    child_to_parent,
                    label_to_index,
                )
                results.append((batch["position"][idx].item(), processed_item))

    # Sort results by position
    results.sort(key=lambda x: x[0])

    # Save results to temporary file
    temp_output_path = f"{cfg.output_path}.temp_{rank}"
    with open(temp_output_path, "wb") as out_file:
        cctx = zstd.ZstdCompressor(level=1)
        with cctx.stream_writer(out_file) as writer:
            for pos, item in results:
                output_item = {"original_position": pos, "data": item}
                writer.write((json.dumps(output_item) + "\n").encode("utf-8"))

    end_time = time.time()
    print(f"GPU {rank}: Finished processing in {end_time - start_time:.2f} seconds")

    # Cleanup and merge results
    dist.barrier()
    if rank == 0:
        print("Combining results from all GPUs...")
        temp_files = [f"{cfg.output_path}.temp_{r}" for r in range(world_size)]
        merge_ordered_files(temp_files, cfg.output_path)
        print("Results combined successfully!")

        # Clean up temp files and preprocessed data
        for temp_file in temp_files:
            os.remove(temp_file)
        for preprocess_file in [
            f"{preprocess_dir}/preprocessed_{r}.pt" for r in range(world_size)
        ]:
            os.remove(preprocess_file)

    cleanup()


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
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

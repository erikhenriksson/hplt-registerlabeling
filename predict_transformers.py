import os
import json
import time
import zstandard as zstd
import io
import torch
from tqdm.auto import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import sigmoid
from torch.utils.data import Dataset, DataLoader
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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class PreprocessedDataset(Dataset):
    def __init__(self, encodings, original_items, positions):
        self.encodings = encodings
        self.original_items = original_items
        self.positions = positions

        # Calculate lengths for dynamic batching
        self.lengths = [len(ids) for ids in self.encodings["input_ids"]]

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "original_item": self.original_items[idx],
            "position": self.positions[idx],
            "length": self.lengths[idx],
        }

    def __len__(self):
        return len(self.positions)


def custom_collate_fn(batch):
    """Custom collate function for dynamic batching"""
    # Sort batch by length
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    # Separate features that need padding from other features
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Get max length in this batch
    max_length = max(len(ids) for ids in input_ids)

    # Pad sequences in this batch
    padded_input_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
    padded_attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        padded_input_ids[i, : len(ids)] = ids
        padded_attention_mask[i, : len(mask)] = mask

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "original_item": [item["original_item"] for item in batch],
        "position": torch.tensor([item["position"] for item in batch]),
    }


def length_bucketed_sampler(lengths, batch_size):
    """Create batches of similar lengths"""
    # Sort indices by length
    indices = list(range(len(lengths)))
    indices.sort(key=lambda i: lengths[i])

    # Create batches with similar lengths
    batches = []
    for i in range(0, len(indices), batch_size):
        batches.append(indices[i : i + batch_size])

    return batches


def process_labels(
    item_data, prob, pred_label, labels_all, child_to_parent, label_to_index
):
    """Process labels for an item"""
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


def preprocess_file(input_path, output_dir, rank, world_size, tokenizer):
    """Preprocess the zst file and save tokenized data for each GPU"""
    os.makedirs(output_dir, exist_ok=True)

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


def merge_ordered_files(temp_files, output_path, buffer_size=8 * 1024 * 1024):
    """Merge multiple ordered files while maintaining global order"""
    active_files = []
    try:
        # Open all temp files
        for temp_file in temp_files:
            f = open(temp_file, "rb")
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(f)
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            active_files.append((f, reader, text_reader))

        with open(output_path, "wb", buffering=buffer_size) as final_out:
            cctx = zstd.ZstdCompressor(level=1)
            with cctx.stream_writer(final_out) as final_writer:
                current_pos = 0
                while True:
                    next_item = None
                    next_file_idx = -1

                    # Find the next item in sequence
                    for idx, (_, _, reader) in enumerate(active_files):
                        line = reader.readline()
                        if not line:
                            continue

                        try:
                            item = json.loads(line)
                            if (
                                next_item is None
                                or item["original_position"]
                                < next_item["original_position"]
                            ):
                                next_item = item
                                next_file_idx = idx
                        except json.JSONDecodeError:
                            continue

                    if next_item is None:
                        break

                    # Write the next item
                    final_writer.write((json.dumps(next_item) + "\n").encode("utf-8"))
                    current_pos += 1

    finally:
        # Clean up file handles
        for f, reader, text_reader in active_files:
            text_reader.close()
            reader.close()
            f.close()


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Load model for this GPU
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Use tokenizer from config
    tokenizer = cfg.tokenizer

    # Stage 1: Preprocess data if not already done
    preprocess_file_path = f"{cfg.preprocess_dir}/preprocessed_{rank}.pt"
    if not os.path.exists(preprocess_file_path):
        preprocess_file(cfg.input_path, cfg.preprocess_dir, rank, world_size, tokenizer)

    # Stage 2: Load preprocessed data and process
    print(f"GPU {rank}: Loading preprocessed data...")
    data = torch.load(preprocess_file_path)
    dataset = PreprocessedDataset(
        data["encodings"], data["original_items"], data["positions"]
    )

    # Create batches based on sequence lengths
    batch_sampler = length_bucketed_sampler(dataset.lengths, cfg.batch_size)

    # Create dataloader with custom collation and batch sampler
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )

    # Process batches
    results = []
    start_time = time.time()

    print(f"GPU {rank}: Starting inference...")
    model.eval()

    # Track padding efficiency
    total_tokens = 0
    total_padding = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=rank != 0):
            # Move inputs to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Calculate padding efficiency for this batch
            batch_total = input_ids.numel()
            batch_padding = (attention_mask == 0).sum().item()
            total_tokens += batch_total
            total_padding += batch_padding

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

    # Print padding efficiency stats
    padding_percentage = (total_padding / total_tokens) * 100
    print(f"GPU {rank}: Padding percentage: {padding_percentage:.2f}%")

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
        for preprocessed_data_file in [
            f"{cfg.preprocess_dir}/preprocessed_{r}.pt" for r in range(world_size)
        ]:
            os.remove(preprocessed_data_file)

    cleanup()


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model", default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/en/1_sample.jsonl.zst")
    parser.add_argument(
        "--output_path", default="data/en/1_sample_register_labels.jsonl.zst"
    )
    parser.add_argument("--preprocess_dir", default="preprocessed_data")
    cfg = parser.parse_args()

    # Create preprocessing directory
    os.makedirs(cfg.preprocess_dir, exist_ok=True)

    # Load tokenizer once in the main process
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    cfg.tokenizer = tokenizer

    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")
    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

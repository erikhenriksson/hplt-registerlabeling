import os

os.environ["HF_HOME"] = ".hf/hf_home"
import json
import time
import zstandard as zstd
import io
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import sigmoid
from itertools import islice
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast

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


def local_data(file_path, rank, world_size, total_lines):
    """Optimized data loading with buffering"""
    chunk_size = total_lines // world_size
    start_line = rank * chunk_size
    end_line = start_line + chunk_size if rank != world_size - 1 else total_lines

    buffer = []
    BUFFER_SIZE = 1000  # Adjust based on memory availability

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

            if buffer:  # Yield remaining items
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
    """Optimized label processing"""
    labels_indexes = pred_label.tolist()

    # Use set for faster lookups
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
    total_lines,
    batch_size=64,
    max_batch_length=12800,
):
    data_iterator = local_data(input_path, rank, world_size, total_lines)
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    processed_count = 0

    # Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    while True:
        large_batch = list(islice(data_iterator, max_batch_length))
        if not large_batch:
            break

        text_data = [(idx, json.loads(line)) for idx, line in enumerate(large_batch)]
        texts = [item[1]["text"] for item in text_data]

        encodings = tokenizer(
            texts, truncation=True, max_length=512, padding=False, return_tensors=None
        )

        lengths = [len(ids) for ids in encodings["input_ids"]]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        processed_results = []

        for i in range(0, len(sorted_indices), batch_size):
            batch_indices = sorted_indices[i : i + batch_size]

            batch_items = [text_data[idx][1] for idx in batch_indices]
            batch_encodings = {
                key: [encodings[key][idx] for idx in batch_indices]
                for key in encodings.keys()
            }

            batch_tokens = tokenizer.pad(
                batch_encodings, padding=True, return_tensors="pt"
            ).to(device)

            # Use autocast for mixed precision
            with torch.no_grad(), autocast(dtype=torch.float16):
                outputs = model(**batch_tokens)
                probabilities = sigmoid(outputs.logits)
                predicted_labels = (probabilities > 0.5).int()

            # Convert back to float32 for CPU operations
            probabilities = probabilities.float()

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

            if processed_count % 1000 == 0:
                print(f"GPU {rank}: Processed {processed_count} items")

        processed_results.sort(key=lambda x: x[0])
        yield [item for _, item in processed_results]


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        total_lines = count_lines_in_zst(cfg.input_path)
        print(f"Total lines to process: {total_lines}")
    else:
        total_lines = None

    total_lines = torch.tensor(total_lines if total_lines is not None else 0).to(device)
    dist.broadcast(total_lines, src=0)
    total_lines = total_lines.item()

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    temp_output_path = f"{cfg.output_path}.temp_{rank}"

    start_time = time.time()
    print(f"GPU {rank}: Starting processing")

    WRITE_BUFFER_SIZE = 1024 * 1024  # 1MB buffer
    with open(temp_output_path, "wb", buffering=WRITE_BUFFER_SIZE) as out_file:
        cctx = zstd.ZstdCompressor(level=3)
        with cctx.stream_writer(out_file) as writer:
            buffer = []
            for processed_batch in batch_process(
                rank,
                world_size,
                model,
                tokenizer,
                cfg.input_path,
                total_lines,
                cfg.batch_size,
                cfg.max_batch_length,
            ):
                buffer.extend(processed_batch)
                if len(buffer) >= 1000:
                    lines = [json.dumps(item) + "\n" for item in buffer]
                    writer.write("".join(lines).encode("utf-8"))
                    buffer = []

            if buffer:
                lines = [json.dumps(item) + "\n" for item in buffer]
                writer.write("".join(lines).encode("utf-8"))

    end_time = time.time()
    print(f"GPU {rank}: Finished processing in {end_time - start_time:.2f} seconds")

    dist.barrier()

    if rank == 0:
        print("Combining results from all GPUs...")
        with open(cfg.output_path, "wb", buffering=WRITE_BUFFER_SIZE) as final_out:
            cctx = zstd.ZstdCompressor(level=3)
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batch_length", type=int, default=1000)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/en/1_sample.jsonl.zst")
    parser.add_argument(
        "--output_path", default="data/en/1_sample_register_labels.jsonl.zst"
    )
    # Add mixed precision flag
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision (FP16) inference",
    )
    cfg = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")
    print("Mixed precision enabled" if cfg.mixed_precision else "Using full precision")
    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

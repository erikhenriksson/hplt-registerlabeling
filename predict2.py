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

# Labels structure remains the same
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
    """Modified to return only the chunk for this rank"""
    chunk_size = total_lines // world_size
    start_line = rank * chunk_size
    end_line = start_line + chunk_size if rank != world_size - 1 else total_lines

    current_line = 0
    with open(file_path, "rb") as file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(file) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                if start_line <= current_line < end_line:
                    yield line
                elif current_line >= end_line:
                    break
                current_line += 1


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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

    # Add a counter for this rank
    total_processed = 0

    print(f"GPU {rank}: Starting processing")  # Add initial log

    while True:
        large_batch = list(islice(data_iterator, max_batch_length))
        if not large_batch:
            break  # End of data

        print(f"GPU {rank}: Got batch of size {len(large_batch)}")  # Log batch size

        text_data = [(idx, json.loads(line)) for idx, line in enumerate(large_batch)]
        text_data = [
            (idx, item, tokenizer(item["text"], truncation=True, max_length=512))
            for idx, item in text_data
        ]

        text_data.sort(key=lambda x: len(x[2]["input_ids"]))

        processed_results = []
        for i in range(0, len(text_data), batch_size):
            print(
                f"GPU {rank}: Processing mini-batch {i//batch_size} of size {min(batch_size, len(text_data)-i)}"
            )
            batch = text_data[i : i + batch_size]
            batch_indices = [item[0] for item in batch]
            batch_items = [item[1] for item in batch]
            batch_tokens = [item[2] for item in batch]

            # Log GPU memory usage periodically
            if i % (batch_size * 10) == 0:
                memory_allocated = torch.cuda.memory_allocated(device=device) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(device=device) / 1024**2
                print(
                    f"GPU {rank}: Memory allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB"
                )

            batch_tokens_padded = tokenizer.pad(
                batch_tokens, padding=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**batch_tokens_padded)
                probabilities = sigmoid(outputs.logits)
                predicted_labels = (probabilities > 0.5).int()

            for idx, item_data, prob, pred_label in zip(
                batch_indices, batch_items, probabilities, predicted_labels
            ):
                labels_indexes = pred_label.tolist()
                labeled_registers = []

                for i, label in enumerate(labels_all):
                    if labels_indexes[i] == 1:
                        if label in child_to_parent:
                            parent_label = child_to_parent[label]
                            labels_indexes[label_to_index[parent_label]] = 1

                labeled_registers = [
                    labels_all[i]
                    for i, active in enumerate(labels_indexes)
                    if active == 1
                ]
                item_data["registers"] = labeled_registers
                item_data["register_probabilities"] = [
                    round(float(p), 4) for p in prob.tolist()
                ]

                processed_results.append((idx, item_data))
                total_processed += 1

        processed_results.sort(key=lambda x: x[0])
        print(f"GPU {rank}: Total processed so far: {total_processed}")  # Log progress
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

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    # Create temporary file for this rank
    temp_output_path = f"{cfg.output_path}.temp_{rank}"

    start_time = time.time()
    print(f"GPU {rank}: Starting processing")

    # Process and save to temporary file
    with open(temp_output_path, "wb") as out_file:
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(out_file) as writer:
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
                for item in processed_batch:
                    line = json.dumps(item) + "\n"
                    writer.write(line.encode("utf-8"))

    end_time = time.time()
    print(f"GPU {rank}: Finished processing in {end_time - start_time:.2f} seconds")

    # Make sure all processes have finished writing their temp files
    dist.barrier()

    # Only rank 0 combines the results
    if rank == 0:
        print("Combining results from all GPUs...")
        with open(cfg.output_path, "wb") as final_out:
            cctx = zstd.ZstdCompressor()
            with cctx.stream_writer(final_out) as final_writer:
                # Read and write from each temp file in order
                for r in range(world_size):
                    temp_file = f"{cfg.output_path}.temp_{r}"
                    with open(temp_file, "rb") as temp_in:
                        dctx = zstd.ZstdDecompressor()
                        with dctx.stream_reader(temp_in) as reader:
                            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                            for line in text_reader:
                                final_writer.write(line.encode("utf-8"))
                    # Remove temp file
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
    cfg = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")
    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

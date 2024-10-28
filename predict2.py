import os
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

# Define input and output paths
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


# Data reading function for .zst files
def local_data(file_path):
    with open(file_path, "rb") as file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(file) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                yield line


# Distributed setup and cleanup functions
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# Batched processing function
def batch_process(
    rank, model, tokenizer, input_path, batch_size=64, max_batch_length=12800
):
    data_iterator = local_data(input_path)
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    while True:
        large_batch = list(islice(data_iterator, max_batch_length))
        if not large_batch:
            break  # End of data

        text_data = [(idx, json.loads(line)) for idx, line in enumerate(large_batch)]
        text_data = [
            (idx, item, tokenizer(item["text"], truncation=True, max_length=512))
            for idx, item in text_data
        ]

        text_data.sort(key=lambda x: len(x[2]["input_ids"]))

        processed_results = []
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i : i + batch_size]
            batch_indices = [item[0] for item in batch]
            batch_items = [item[1] for item in batch]
            batch_tokens = [item[2] for item in batch]

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

        processed_results.sort(key=lambda x: x[0])
        yield [item for _, item in processed_results]


# Process and save function
def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    start_time = time.time()
    with open(cfg.output_path, "wb") as out_file:
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(out_file) as writer:
            for processed_batch in batch_process(
                rank,
                model,
                tokenizer,
                cfg.input_path,
                cfg.batch_size,
                cfg.max_batch_length,
            ):
                for item in processed_batch:
                    line = json.dumps(item) + "\n"
                    writer.write(line.encode("utf-8"))

    end_time = time.time()
    if rank == 0:
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

    cleanup()


# Main function to run with torch.multiprocessing.spawn
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
    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

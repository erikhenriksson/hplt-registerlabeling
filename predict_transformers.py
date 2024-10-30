import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import json
import zstandard as zstd
from argparse import ArgumentParser
import os
from typing import List, Dict, Any, Iterator, Tuple
import numpy as np
from torch.cuda.amp import autocast
import io
from collections import defaultdict


def read_zst_chunks(file_path: str, chunk_size: int = 10000) -> Iterator[List[Dict]]:
    """Stream data from zst file in chunks."""
    chunk = []
    chunk_start_idx = 0  # Track starting index of each chunk
    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line_idx, line in enumerate(text_stream):
                if len(chunk) >= chunk_size:
                    yield chunk_start_idx, chunk
                    chunk = []
                    chunk_start_idx = line_idx
                item = json.loads(line)
                chunk.append(item)
    if chunk:  # Don't forget the last chunk
        yield chunk_start_idx, chunk


def tokenize_and_sort(
    texts: List[Dict], chunk_start_idx: int, tokenizer
) -> Tuple[List[int], dict]:
    """Tokenize texts and return sorted indices and encodings."""
    # Tokenize all texts at once
    encodings = tokenizer(
        [item["text"] for item in texts],
        padding=False,  # No padding yet as we're just getting lengths
        truncation=True,
        max_length=512,
        return_tensors=None,  # Return list of token ids
    )

    # Get lengths and create index pairs
    text_lengths = [(i, len(tokens)) for i, tokens in enumerate(encodings["input_ids"])]

    # Sort by length but keep track of original position
    text_lengths.sort(key=lambda x: x[1])
    sorted_indices = [i for i, _ in text_lengths]

    # Add original global index to each item
    for i, idx in enumerate(sorted_indices):
        texts[idx]["original_idx"] = chunk_start_idx + idx

    return sorted_indices, encodings


def create_length_batches(
    texts: List[Dict], sorted_indices: List[int], encodings: dict, batch_size: int
) -> List[Tuple[List[Dict], dict]]:
    """Create batches of similar length texts with their encodings."""
    batches = []
    current_batch_indices = []
    current_batch_texts = []

    for idx in sorted_indices:
        current_batch_indices.append(idx)
        current_batch_texts.append(texts[idx])

        if len(current_batch_indices) >= batch_size:
            batch_encodings = {
                "input_ids": [encodings["input_ids"][i] for i in current_batch_indices],
                "attention_mask": [
                    encodings["attention_mask"][i] for i in current_batch_indices
                ],
            }
            batches.append((current_batch_texts, batch_encodings))
            current_batch_indices = []
            current_batch_texts = []

    if current_batch_texts:
        batch_encodings = {
            "input_ids": [encodings["input_ids"][i] for i in current_batch_indices],
            "attention_mask": [
                encodings["attention_mask"][i] for i in current_batch_indices
            ],
        }
        batches.append((current_batch_texts, batch_encodings))

    return batches


def process_batch(
    batch_texts: List[Dict], batch_encodings: dict, model, device
) -> Tuple[List[Dict], float]:
    """Process a single batch and return results with timing."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in batch_encodings["input_ids"]], batch_first=True
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in batch_encodings["attention_mask"]],
        batch_first=True,
        padding_value=0,
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits)

    probs = probs.float()

    end_time.record()
    torch.cuda.synchronize()
    inference_time = start_time.elapsed_time(end_time)

    probs = probs.cpu().numpy()
    registers = (probs > 0.5).astype(bool)

    results = []
    for item, prob_row, reg_row in zip(batch_texts, probs, registers):
        results.append(
            {
                "id": item["id"],
                "registers": np.where(reg_row)[0].tolist(),
                "probabilities": [f"{p:.4f}" for p in prob_row],
                "original_idx": item["original_idx"],  # Keep track of original position
            }
        )

    return results, inference_time


def save_results(results: List[Dict], output_path: str, mode: str = "w"):
    """Save results to CSV file in original order."""
    # Sort by original index before saving
    results.sort(key=lambda x: x["original_idx"])

    # Remove the temporary original_idx field
    for result in results:
        del result["original_idx"]

    df = pd.DataFrame(results)
    df.to_csv(output_path, mode=mode, header=(mode == "w"), index=False)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path, num_labels=25, torch_dtype=torch.bfloat16
    ).to(rank)

    model = DDP(model, device_ids=[rank])
    model.eval()

    total_inference_time = 0
    total_batches = 0
    total_examples = 0

    chunk_results = defaultdict(list)
    first_chunk = True

    for chunk_idx, (chunk_start_idx, chunk) in enumerate(
        read_zst_chunks(cfg.input_path, chunk_size=cfg.chunk_size)
    ):
        if rank == 0:
            print(f"Processing chunk {chunk_idx + 1}...")

        sorted_indices, encodings = tokenize_and_sort(
            chunk, chunk_start_idx, cfg.tokenizer
        )
        batches = create_length_batches(
            chunk, sorted_indices, encodings, cfg.batch_size
        )
        rank_batches = batches[rank::world_size]

        for batch_idx, (batch_texts, batch_encodings) in enumerate(rank_batches):
            results, inference_time = process_batch(
                batch_texts, batch_encodings, model, rank
            )

            total_inference_time += inference_time
            total_batches += 1
            total_examples += len(batch_texts)

            chunk_results[rank].extend(results)

        gathered_results = [None] * world_size if rank == 0 else None
        dist.gather_object(chunk_results[rank], gathered_results, dst=0)

        if rank == 0:
            all_results = []
            for result_list in gathered_results:
                if result_list:
                    all_results.extend(result_list)

            save_results(all_results, cfg.output_path, mode="w" if first_chunk else "a")
            first_chunk = False

        chunk_results[rank] = []
        dist.barrier()

    if rank == 0:
        print("\nFinal Timing Statistics:")
        print(f"Total Inference Time: {total_inference_time/1000:.2f} seconds")
        print(f"Average Time per Batch: {total_inference_time/total_batches:.2f}ms")
        print(f"Average Time per Example: {total_inference_time/total_examples:.2f}ms")
        print(f"Total Examples Processed: {total_examples}")
        print(
            f"Overall Throughput: {1000 * total_examples / total_inference_time:.2f} examples/sec"
        )

    cleanup()


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model", default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/input.jsonl.zst")
    parser.add_argument("--output_path", default="output.csv")

    cfg = parser.parse_args()

    # Load tokenizer once in the main process
    global tokenizer  # Make tokenizer globally available
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    cfg.tokenizer = tokenizer

    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")

    # Adjust batch size for multi-GPU
    if world_size > 1:
        cfg.batch_size = cfg.batch_size * world_size
        print(f"Adjusted batch size to {cfg.batch_size} for multi-GPU processing")

    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

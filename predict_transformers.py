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
    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
                chunk.append(json.loads(line))
    if chunk:  # Don't forget the last chunk
        yield chunk


def tokenize_and_sort(texts: List[Dict], tokenizer) -> Tuple[List[int], dict]:
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

    # Sort by length
    text_lengths.sort(key=lambda x: x[1])
    sorted_indices = [i for i, _ in text_lengths]

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
            # Create batch encodings
            batch_encodings = {
                "input_ids": [encodings["input_ids"][i] for i in current_batch_indices],
                "attention_mask": [
                    encodings["attention_mask"][i] for i in current_batch_indices
                ],
            }
            batches.append((current_batch_texts, batch_encodings))
            current_batch_indices = []
            current_batch_texts = []

    if current_batch_texts:  # Don't forget the last batch
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
) -> List[Dict]:
    """Process a single batch and return results."""
    # Pad and convert to tensors
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in batch_encodings["input_ids"]], batch_first=True
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in batch_encodings["attention_mask"]],
        batch_first=True,
        padding_value=0,
    )

    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Get predictions
    # with torch.no_grad(), autocast(dtype=torch.bfloat16):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Convert to probabilities
    probs = torch.sigmoid(outputs.logits.float()).cpu().numpy()
    registers = (probs > 0.5).astype(bool)

    # Format results
    results = []
    for item, prob_row, reg_row in zip(batch_texts, probs, registers):
        results.append(
            {
                "id": item["id"],
                "registers": np.where(reg_row)[0].tolist(),
                "probabilities": [f"{p:.4f}" for p in prob_row],
            }
        )

    return results


def save_results(results: List[Dict], output_path: str, mode: str = "w"):
    """Save results to CSV file, either creating new file or appending."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, mode=mode, header=(mode == "w"), index=False)


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def process_and_save_ddp(rank, cfg, world_size):
    """Main processing function for each GPU."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path,
        # num_labels=25,  # Assuming 25 registers
        torch_dtype=torch.bfloat16,
    ).to(rank)

    model = DDP(model, device_ids=[rank])
    model.eval()

    # Process data in chunks
    chunk_results = defaultdict(list)
    first_chunk = True

    for chunk_idx, chunk in enumerate(
        read_zst_chunks(cfg.input_path, chunk_size=cfg.chunk_size)
    ):
        if rank == 0:
            print(f"Processing chunk {chunk_idx + 1}...")

        # Tokenize and sort chunk
        sorted_indices, encodings = tokenize_and_sort(chunk, cfg.tokenizer)

        # Create length-based batches with their encodings
        batches = create_length_batches(
            chunk, sorted_indices, encodings, cfg.batch_size
        )

        # Distribute batches across GPUs
        rank_batches = batches[rank::world_size]

        # Process batches assigned to this rank
        for batch_idx, (batch_texts, batch_encodings) in enumerate(rank_batches):
            if rank == 0 and batch_idx % 10 == 0:
                print(f"GPU 0: Processing batch {batch_idx + 1}/{len(rank_batches)}")

            results = process_batch(batch_texts, batch_encodings, model, rank)
            chunk_results[rank].extend(results)

        # Gather results from all ranks
        gathered_results = [None] * world_size if rank == 0 else None
        dist.gather_object(chunk_results[rank], gathered_results, dst=0)

        # Save results (rank 0 only)
        if rank == 0:
            all_results = []
            for result_list in gathered_results:
                all_results.extend(result_list)

            # Save results
            save_results(all_results, cfg.output_path, mode="w" if first_chunk else "a")
            print(f"Saved results for chunk {chunk_idx + 1}")
            first_chunk = False

        # Clear results for this chunk
        chunk_results[rank] = []

        # Synchronize processes before next chunk
        dist.barrier()

    cleanup()


def main():
    parser = ArgumentParser(
        description="Process large text files with register classification"
    )
    parser.add_argument(
        "--base_model",
        default="xlm-roberta-base",
        help="Base model to use for tokenizer",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of examples to process at once",
    )
    parser.add_argument(
        "--model_path",
        default="models/xlm-roberta-base",
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--input_path", default="data/input.jsonl.zst", help="Path to input zst file"
    )
    parser.add_argument(
        "--output_path", default="output.csv", help="Path to output CSV file"
    )

    cfg = parser.parse_args()

    # Load tokenizer once in the main process
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    cfg.tokenizer = tokenizer

    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")

    # Launch DDP processes
    print("Starting distributed processing...")
    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)

    print("Processing complete!")


if __name__ == "__main__":
    main()

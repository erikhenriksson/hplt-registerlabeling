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
import gc


def read_zst_chunks(file_path: str, chunk_size: int = 1000) -> Iterator[List[Dict]]:
    """Stream data from zst file in smaller chunks to manage memory."""
    chunk = []
    chunk_start_idx = 0
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
    if chunk:
        yield chunk_start_idx, chunk


def parallel_tokenize(
    texts: List[Dict], tokenizer, batch_size: int = 1000, reset_interval: int = 10
) -> List[Dict]:
    """Tokenize texts in smaller batches with periodic tokenizer reset to reduce memory usage."""
    input_ids_list = []
    attention_mask_list = []

    for batch_num, i in enumerate(range(0, len(texts), batch_size)):
        batch_texts = [item["text"] for item in texts[i : i + batch_size]]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )

        input_ids_list.append(encodings["input_ids"].cpu())
        attention_mask_list.append(encodings["attention_mask"].cpu())

        del batch_texts, encodings
        gc.collect()
        torch.cuda.empty_cache()

        if (batch_num + 1) % reset_interval == 0:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path)

    # Concatenate and immediately convert to CPU tensors
    all_input_ids = torch.cat(input_ids_list, dim=0)
    all_attention_mask = torch.cat(attention_mask_list, dim=0)

    # Clear intermediate lists
    del input_ids_list, attention_mask_list
    gc.collect()

    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask}


def save_results(results: List[Dict], output_path: str, mode: str = "a"):
    """Save results using the original order, optimized with file appending."""
    try:
        results.sort(key=lambda x: x["original_idx"])
        for result in results:
            del result["original_idx"]

        # Use chunks to write to file to avoid memory spikes
        chunk_size = 1000
        for i in range(0, len(results), chunk_size):
            chunk = results[i : i + chunk_size]
            df = pd.DataFrame(chunk)
            df.to_csv(
                output_path, mode=mode, header=(mode == "w" and i == 0), index=False
            )
            del df
            gc.collect()
    finally:
        # Ensure results are cleared
        results.clear()
        gc.collect()


def process_batch(
    batch_texts: List[Dict], batch_encodings: dict, model, device
) -> Tuple[List[Dict], float]:
    """Process a batch with padding optimization and track inference timing."""
    try:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in batch_encodings["input_ids"]], batch_first=True
        ).to(device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in batch_encodings["attention_mask"]],
            batch_first=True,
            padding_value=0,
        ).to(device)

        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()

        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).float()

        end_time.record()
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)

        # Move to CPU and convert to numpy immediately
        probs = probs.cpu().numpy()
        registers = (probs > 0.5).astype(bool)

        results = [
            {
                "id": item["id"],
                "registers": np.where(reg_row)[0].tolist(),
                "probabilities": [f"{p:.4f}" for p in prob_row],
                "original_idx": item["original_idx"],
            }
            for item, prob_row, reg_row in zip(batch_texts, probs, registers)
        ]

        return results, inference_time

    finally:
        # Explicit cleanup
        del input_ids, attention_mask, outputs, probs, registers
        torch.cuda.empty_cache()
        gc.collect()


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

    if rank == 0:
        with open(cfg.output_path, "w") as f:
            f.write("id,registers,probabilities\n")

    try:
        for chunk_idx, (chunk_start_idx, chunk) in enumerate(
            read_zst_chunks(cfg.input_path, chunk_size=cfg.chunk_size)
        ):
            # Process chunk
            sorted_indices, encodings = tokenize_and_sort(
                chunk, chunk_start_idx, cfg.tokenizer
            )
            batches = create_length_batches(
                chunk, sorted_indices, encodings, cfg.batch_size
            )

            # Clear encodings after batch creation
            del encodings
            gc.collect()

            # Process rank-specific batches
            rank_batches = batches[rank::world_size]
            chunk_results = []

            for batch_texts, batch_encodings in rank_batches:
                results, inference_time = process_batch(
                    batch_texts, batch_encodings, model, rank
                )

                total_inference_time += inference_time
                total_batches += 1
                total_examples += len(batch_texts)

                chunk_results.extend(results)
                del results
                gc.collect()

            # Gather results from all ranks
            if rank == 0:
                gathered_results = [None] * world_size
            else:
                gathered_results = None

            dist.gather_object(chunk_results, gathered_results, dst=0)

            # Clear chunk results after gathering
            del chunk_results
            gc.collect()

            # Save results on rank 0
            if rank == 0:
                try:
                    all_results = []
                    for result_list in gathered_results:
                        if result_list:
                            all_results.extend(result_list)
                            result_list.clear()  # Clear each result list after extending

                    save_results(all_results, cfg.output_path, mode="a")
                finally:
                    # Ensure cleanup of gathered results
                    del all_results, gathered_results
                    gc.collect()

            # Synchronize before next chunk
            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()

            if rank == 0 and chunk_idx % 10 == 0:
                batch_throughput = 1000 * len(batch_texts) / inference_time
                print(
                    f"Processed {total_examples} examples. "
                    f"Batch {chunk_idx + 1} throughput: {batch_throughput:.2f} examples/sec"
                )

    finally:
        cleanup()
        if rank == 0:
            print("\nFinal Timing Statistics:")
            print(f"Total Inference Time: {total_inference_time / 1000:.2f} seconds")
            print(
                f"Average Time per Batch: {total_inference_time / total_batches:.2f}ms"
            )
            print(
                f"Average Time per Example: {total_inference_time / total_examples:.2f}ms"
            )
            print(f"Total Examples Processed: {total_examples}")
            print(
                f"Overall Throughput: {1000 * total_examples / total_inference_time:.2f} examples/sec"
            )


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model", default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/input.jsonl.zst")
    parser.add_argument("--output_path", default="output.csv")

    cfg = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    cfg.tokenizer = tokenizer

    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")

    if world_size > 1:
        cfg.batch_size = cfg.batch_size * world_size
        print(f"Adjusted batch size to {cfg.batch_size} for multi-GPU processing")

    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

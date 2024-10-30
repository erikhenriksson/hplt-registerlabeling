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
from torch.utils.data import Dataset, DataLoader
import time


def read_zst_chunks(
    file_path: str, chunk_size: int = 1000
) -> Iterator[Tuple[int, List[Dict]]]:
    """Stream data from zst file in smaller chunks to manage memory."""
    chunk = []
    start_idx = 0
    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for idx, line in enumerate(text_stream):
                item = json.loads(line)
                item["original_idx"] = start_idx + idx  # Assign original index
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    yield start_idx, chunk
                    start_idx += len(chunk)
                    chunk = []
            if chunk:
                yield start_idx, chunk


class ChunkDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.lengths = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        """Calculate sequence lengths for length-based batching."""
        for idx, item in enumerate(self.data):
            encoding = self.tokenizer(
                item["text"],
                truncation=True,
                max_length=512,
                padding=False,
                return_length=True,
            )
            seq_length = encoding["length"][0]
            self.lengths.append(seq_length)

        # Sort data by sequence length
        self.sorted_indices = np.argsort(self.lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sorted_idx = self.sorted_indices[idx]  # Access items in sorted order
        item = self.data[sorted_idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["id"] = item["id"]
        encoding["original_idx"] = item["original_idx"]  # Include original index
        return encoding


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    ids = [item["id"] for item in batch]
    original_indices = [item["original_idx"] for item in batch]

    # Dynamic padding
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ids": ids,
        "original_indices": original_indices,
    }


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
        cfg.model_path, num_labels=25, torch_dtype=torch.float16
    ).to(rank)

    model = DDP(model, device_ids=[rank])
    model.eval()

    total_batches = 0
    total_examples = 0

    if rank == 0:
        with open(cfg.output_path, "w") as f:
            f.write("id,registers,probabilities\n")

    start_time = time.time()

    for chunk_idx, (chunk_start_idx, chunk) in enumerate(
        read_zst_chunks(cfg.input_path, chunk_size=cfg.chunk_size)
    ):
        # Create dataset and dataloader for the current chunk
        dataset = ChunkDataset(chunk, cfg.tokenizer)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=4,  # Adjust based on your CPU cores
            pin_memory=True,
            collate_fn=collate_fn,
        )

        chunk_results = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(rank, non_blocking=True)
            attention_mask = batch["attention_mask"].to(rank, non_blocking=True)
            ids = batch["ids"]
            original_indices = batch["original_indices"]

            with torch.no_grad(), autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)

            probs = probs.float().cpu().numpy()
            registers = (probs > 0.5).astype(bool)

            batch_results = []
            for idx, (id_, prob_row, reg_row, orig_idx) in enumerate(
                zip(ids, probs, registers, original_indices)
            ):
                batch_results.append(
                    {
                        "id": id_,
                        "registers": np.where(reg_row)[0].tolist(),
                        "probabilities": [f"{p:.4f}" for p in prob_row],
                        "original_idx": orig_idx,
                    }
                )

            chunk_results.extend(batch_results)
            total_batches += 1
            total_examples += len(ids)

        # Gather results from all ranks
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, chunk_results)

        if rank == 0:
            all_chunk_results = []
            for res in gathered_results:
                all_chunk_results.extend(res)
            # Sort results back to original order
            all_chunk_results.sort(key=lambda x: x["original_idx"])

            # Remove original_idx from output
            for item in all_chunk_results:
                del item["original_idx"]

            # Save results
            df = pd.DataFrame(all_chunk_results)
            df.to_csv(cfg.output_path, mode="a", header=False, index=False)

            elapsed_time = time.time() - start_time
            batch_throughput = total_examples / elapsed_time
            if chunk_idx % 10 == 0:
                print(
                    f"Processed {total_examples} examples. "
                    f"Chunk {chunk_idx + 1} throughput: {batch_throughput:.2f} examples/sec"
                )

    if rank == 0:
        total_inference_time = time.time() - start_time
        print("\nFinal Timing Statistics:")
        print(f"Total Inference Time: {total_inference_time:.2f} seconds")
        print(f"Total Examples Processed: {total_examples}")
        print(
            f"Overall Throughput: {total_examples / total_inference_time:.2f} examples/sec"
        )

    cleanup()


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model", default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=10000)  # Adjust as needed
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/input.jsonl.zst")
    parser.add_argument("--output_path", default="output.csv")

    cfg = parser.parse_args()

    # Load tokenizer once in the main process
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    cfg.tokenizer = tokenizer

    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")

    # Adjust batch size for multi-GPU
    if world_size > 1:
        cfg.batch_size = cfg.batch_size  # Keep batch size per GPU
        print(f"Batch size per GPU: {cfg.batch_size}")

    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

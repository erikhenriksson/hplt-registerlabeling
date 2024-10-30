import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import json
import zstandard as zstd
from argparse import ArgumentParser
import os
from typing import List, Dict, Any
import numpy as np
from itertools import islice
from torch.cuda.amp import autocast
import io


def collate_fn_creator(tokenizer):
    def collate_fn(batch: List[Dict[str, Any]]):
        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]

        # Tokenize
        encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        return {
            "ids": ids,
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    return collate_fn


class StreamingJSONLDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, chunk_size=1000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.data = []
        self.current_position = 0

        # Get total number of lines
        with open(file_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                self.total_lines = sum(1 for _ in text_stream)

        # Load first chunk
        self._load_next_chunk()

    def _load_next_chunk(self):
        self.data = []
        with open(self.file_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                # Skip to current position
                for _ in islice(text_stream, self.current_position):
                    pass
                # Load chunk_size lines
                for _ in range(self.chunk_size):
                    try:
                        line = next(text_stream)
                        item = json.loads(line)
                        self.data.append(item)
                    except StopIteration:
                        break

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        chunk_idx = idx - self.current_position
        if chunk_idx >= len(self.data) or chunk_idx < 0:
            self.current_position = (idx // self.chunk_size) * self.chunk_size
            self._load_next_chunk()
            chunk_idx = idx - self.current_position

        item = self.data[chunk_idx]
        return {"id": item["id"], "text": item["text"]}


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class BatchSampler:
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path,
        num_labels=25,  # Assuming 25 registers
        torch_dtype=torch.bfloat16,
    ).to(rank)

    model = DDP(model, device_ids=[rank])
    model.eval()

    # Create dataset and dataloader
    dataset = StreamingJSONLDataset(cfg.input_path, cfg.tokenizer)

    # Create sampler for DDP
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # Keep order for proper gathering
    )

    # Create dataloader with custom batch sampler
    batch_sampler = BatchSampler(sampler, cfg.batch_size)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn_creator(cfg.tokenizer),
        num_workers=4,
    )

    # Process data
    results = []
    with torch.no_grad():
        for batch in dataloader:
            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"].to(rank),
                    attention_mask=batch["attention_mask"].to(rank),
                )

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            # Get registers above threshold
            registers = (probs > 0.5).astype(bool)

            # Format results
            for idx, (id_, prob_row, reg_row) in enumerate(
                zip(batch["ids"], probs, registers)
            ):
                results.append(
                    {
                        "id": id_,
                        "registers": np.where(reg_row)[0].tolist(),
                        "probabilities": [f"{p:.4f}" for p in prob_row],
                    }
                )

    # Gather results from all processes
    gathered_results = [None] * world_size if rank == 0 else None
    dist.gather_object(results, gathered_results, dst=0)

    # Save results (only on rank 0)
    if rank == 0:
        all_results = []
        for result_list in gathered_results:
            all_results.extend(result_list)
        df = pd.DataFrame(all_results)
        df.to_csv(cfg.output_path, index=False)

    cleanup()


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model", default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/en/1_sample.jsonl.zst")
    parser.add_argument("--output_path", default="output.csv")
    parser.add_argument("--preprocess_dir", default="preprocessed_data")

    cfg = parser.parse_args()

    # Load tokenizer once in the main process
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    cfg.tokenizer = tokenizer

    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Running with {world_size} GPUs")

    # Launch DDP processes
    mp.spawn(process_and_save_ddp, args=(cfg, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

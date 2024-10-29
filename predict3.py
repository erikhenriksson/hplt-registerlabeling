import os
import gc
import psutil
import json
import time
import weakref
import zstandard as zstd
import io
import torch
from typing import Optional, Dict, List
from itertools import islice
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import contextmanager
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import sigmoid

os.environ["HF_HOME"] = ".hf/hf_home"
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


def get_process_memory():
    """Get memory usage info for the current process"""
    process = psutil.Process(os.getpid())
    return {
        "rss": process.memory_info().rss / (1024 * 1024 * 1024),
        "vms": process.memory_info().vms / (1024 * 1024 * 1024),
        "fds": process.num_fds() if hasattr(process, "num_fds") else 0,
        "threads": process.num_threads(),
    }


class OrderedFileReader:
    """Helper class to read and track items from a file with their positions"""

    def __init__(self, file_path):
        self.file = open(file_path, "rb", buffering=16 * 1024 * 1024)
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.file)
        self.text_reader = io.TextIOWrapper(self.reader, encoding="utf-8")
        self.current_item = None
        self.exhausted = False
        self._finalizer = weakref.finalize(self, self.close)
        self._read_next()

    def _read_next(self):
        try:
            line = self.text_reader.readline()
            if not line:
                self.exhausted = True
                self.current_item = None
            else:
                item = json.loads(line)
                self.current_item = (item["original_position"], item["data"])
        except Exception as e:
            print(f"Error reading line: {e}")
            self.exhausted = True
            self.current_item = None

    def get_current(self):
        return self.current_item

    def advance(self):
        self._read_next()

    def close(self):
        self.text_reader.close()
        self.reader.close()
        self.file.close()


class DataIterator:
    def __init__(
        self, file_path: str, rank: int, world_size: int, buffer_size: int = 10000
    ):
        self.file_path = file_path
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self._buffer: List = []
        self._dctx = zstd.ZstdDecompressor()
        self._finalizer = weakref.finalize(self, self._cleanup)
        self._file = None
        self._reader = None
        self._line_count = 0

    def _cleanup(self):
        if hasattr(self, "_reader") and self._reader:
            self._reader.close()
        if hasattr(self, "_file") and self._file:
            self._file.close()
        self._buffer.clear()

    def __iter__(self):
        self._file = open(self.file_path, "rb", buffering=16 * 1024 * 1024)
        reader = self._dctx.stream_reader(self._file)
        self._reader = io.TextIOWrapper(reader, encoding="utf-8")
        return self

    def __next__(self):
        if not self._buffer:
            chunk = []
            for _ in range(self.buffer_size):
                line = self._reader.readline()
                if not line:
                    if not chunk:
                        raise StopIteration
                    break
                if self._line_count % self.world_size == self.rank:
                    chunk.append((self._line_count, line))
                self._line_count += 1
            self._buffer = chunk

        return self._buffer.pop(0)


def process_labels(item_data, prob, pred_label):
    """Optimized label processing"""
    active_labels = {
        labels_all[i] for i, is_active in enumerate(pred_label) if is_active
    }
    active_labels.update(
        child_to_parent[label] for label in active_labels if label in child_to_parent
    )

    item_data["registers"] = list(active_labels)
    item_data["register_probabilities"] = [round(float(p), 4) for p in prob.tolist()]
    return item_data


class BatchProcessor:
    def __init__(self, model, tokenizer, device, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @torch.inference_mode()
    def process_batch(self, texts, positions):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encodings)
        probabilities = sigmoid(outputs.logits)
        predicted_labels = (probabilities > 0.5).int()

        return probabilities, predicted_labels


def merge_ordered_files(temp_files, output_path):
    """Merge multiple ordered files while maintaining global order"""
    readers = [OrderedFileReader(f) for f in temp_files]
    active_readers = len(readers)

    with open(output_path, "wb", buffering=16 * 1024 * 1024) as final_out:
        with zstd.ZstdCompressor(level=1).stream_writer(final_out) as writer:
            buffer = []
            buffer_size = 0
            position = 0

            while active_readers > 0:
                min_pos = float("inf")
                min_idx = -1

                for idx, reader in enumerate(readers):
                    if not reader.exhausted:
                        current = reader.get_current()
                        if current and current[0] < min_pos:
                            min_pos = current[0]
                            min_idx = idx

                if min_idx == -1:
                    break

                assert (
                    min_pos == position
                ), f"Position mismatch: expected {position}, got {min_pos}"

                pos, item = readers[min_idx].get_current()
                readers[min_idx].advance()
                if readers[min_idx].exhausted:
                    active_readers -= 1

                json_str = json.dumps({"original_position": pos, "data": item}) + "\n"
                buffer.append(json_str)
                buffer_size += len(json_str)

                if buffer_size >= 16 * 1024 * 1024:  # 16MB buffer
                    writer.write("".join(buffer).encode("utf-8"))
                    buffer = []
                    buffer_size = 0

                position += 1

            if buffer:
                writer.write("".join(buffer).encode("utf-8"))

    for reader in readers:
        reader.close()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def process_and_save_ddp(rank, cfg, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        processor = BatchProcessor(model, tokenizer, device)

        temp_output_path = f"{cfg.output_path}.temp_{rank}"
        processed_count = 0

        with open(temp_output_path, "wb", buffering=16 * 1024 * 1024) as out_file:
            with zstd.ZstdCompressor(level=1).stream_writer(out_file) as writer:
                buffer = []
                buffer_size = 0

                for batch in DataIterator(cfg.input_path, rank, world_size):
                    items = json.loads(batch[1])
                    text = items["text"]
                    probabilities, predictions = processor.process_batch(
                        [text], [batch[0]]
                    )

                    processed_item = process_labels(
                        items.copy(), probabilities[0], predictions[0]
                    )

                    output = {"original_position": batch[0], "data": processed_item}

                    json_str = json.dumps(output) + "\n"
                    buffer.append(json_str)
                    buffer_size += len(json_str)

                    if buffer_size >= 16 * 1024 * 1024:  # 16MB buffer
                        writer.write("".join(buffer).encode("utf-8"))
                        buffer = []
                        buffer_size = 0

                    processed_count += 1
                    if rank == 0 and processed_count % 10000 == 0:
                        print(f"GPU {rank}: Processed {processed_count} items")

                if buffer:
                    writer.write("".join(buffer).encode("utf-8"))

        dist.barrier()

        if rank == 0:
            print("Merging results from all GPUs...")
            temp_files = [f"{cfg.output_path}.temp_{r}" for r in range(world_size)]
            merge_ordered_files(temp_files, cfg.output_path)

            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    finally:
        cleanup()


def main():
    from argparse import ArgumentParser

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

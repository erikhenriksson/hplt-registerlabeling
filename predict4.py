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


@contextmanager
def manage_memory():
    """Context manager for memory cleanup"""
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


class OrderedFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self._open_resources()
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _open_resources(self):
        """Separate method for opening resources to allow reopening if needed"""
        self.file = open(self.file_path, "rb", buffering=16 * 1024 * 1024)
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.file)
        self.text_reader = io.TextIOWrapper(self.reader, encoding="utf-8")
        self.current_item = None
        self.exhausted = False
        self._read_next()

    def _cleanup(self):
        """Enhanced cleanup"""
        if hasattr(self, "text_reader") and self.text_reader:
            self.text_reader.close()
            self.text_reader = None
        if hasattr(self, "reader") and self.reader:
            self.reader.close()
            self.reader = None
        if hasattr(self, "file") and self.file:
            self.file.close()
            self.file = None
        self.current_item = None
        gc.collect()

    def _read_next(self):
        try:
            line = self.text_reader.readline()
            if not line:
                self.exhausted = True
                self.current_item = None
                return

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
        self._cleanup()


class DataIterator:
    def __init__(
        self, file_path: str, rank: int, world_size: int, buffer_size: int = 5000
    ):
        self.file_path = file_path
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self._buffer: List = []
        self._line_count = 0
        self._finalizer = weakref.finalize(self, self._cleanup)
        self._init_resources()

    def _init_resources(self):
        """Initialize resources with proper error handling"""
        try:
            self._file = open(self.file_path, "rb", buffering=16 * 1024 * 1024)
            self._dctx = zstd.ZstdDecompressor()
            self._reader = self._dctx.stream_reader(self._file)
            self._text_reader = io.TextIOWrapper(self._reader, encoding="utf-8")
        except Exception as e:
            self._cleanup()
            raise e

    def _cleanup(self):
        """Enhanced cleanup with explicit resource handling"""
        if hasattr(self, "_text_reader") and self._text_reader:
            self._text_reader.close()
            self._text_reader = None
        if hasattr(self, "_reader") and self._reader:
            self._reader.close()
            self._reader = None
        if hasattr(self, "_file") and self._file:
            self._file.close()
            self._file = None
        self._buffer.clear()
        gc.collect()

    def __iter__(self):
        self._init_resources()
        return self

    def __next__(self):
        if not self._buffer:
            with manage_memory():
                chunk = []
                for _ in range(self.buffer_size):
                    line = self._text_reader.readline()
                    if not line:
                        if not chunk:
                            self._cleanup()
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
        with manage_memory():
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

            # Detach and move to CPU to prevent GPU memory accumulation
            probabilities = probabilities.detach().cpu()
            predicted_labels = predicted_labels.detach().cpu()

            # Clear CUDA cache after processing
            del encodings
            del outputs
            torch.cuda.empty_cache()

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
        with manage_memory():
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            model = DDP(model, device_ids=[rank], find_unused_parameters=False)
            with open(f"{cfg.model_path}/config.json", "r") as config_file:
                config = json.load(config_file)
            tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))
            processor = BatchProcessor(model, tokenizer, device)
            start_time = time.time()

            if rank == 0:
                print(f"Starting processing on {world_size} GPUs")
            temp_output_path = f"{cfg.output_path}.temp_{rank}"
            processed_count = 0

            # Track initial memory state
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            with open(temp_output_path, "wb", buffering=16 * 1024 * 1024) as out_file:
                with zstd.ZstdCompressor(level=1).stream_writer(out_file) as writer:
                    buffer = []
                    buffer_size = 0

                    for batch in DataIterator(cfg.input_path, rank, world_size):
                        with manage_memory():
                            items = json.loads(batch[1])
                            text = items["text"]
                            probabilities, predictions = processor.process_batch(
                                [text], [batch[0]]
                            )

                            processed_item = process_labels(
                                items.copy(), probabilities[0], predictions[0]
                            )

                            output = {
                                "original_position": batch[0],
                                "data": processed_item,
                            }

                            json_str = json.dumps(output) + "\n"
                            buffer.append(json_str)
                            buffer_size += len(json_str)

                            if buffer_size >= 16 * 1024 * 1024:
                                writer.write("".join(buffer).encode("utf-8"))
                                buffer = []
                                buffer_size = 0

                            processed_count += 1

                            # Memory cleanup every 5000 items
                            if processed_count % 5000 == 0:
                                gc.collect()
                                torch.cuda.empty_cache()

                                if rank == 0:
                                    current_memory = (
                                        psutil.Process().memory_info().rss
                                        / (1024 * 1024)
                                    )
                                    print(
                                        f"GPU {rank}: Processed {processed_count} items. "
                                        f"Memory usage: {current_memory - initial_memory:.2f}MB"
                                    )

                    if buffer:
                        writer.write("".join(buffer).encode("utf-8"))

        dist.barrier()
        end_time = time.time()
        print(f"GPU {rank}: Finished processing in {end_time - start_time:.2f} seconds")

        if rank == 0:
            merge_ordered_files(
                [f"{cfg.output_path}.temp_{r}" for r in range(world_size)],
                cfg.output_path,
            )

            # Clean up temp files
            for r in range(world_size):
                try:
                    os.remove(f"{cfg.output_path}.temp_{r}")
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

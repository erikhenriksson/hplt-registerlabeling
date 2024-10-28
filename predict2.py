import os

os.environ["HF_HOME"] = ".hf/hf_home"
from argparse import ArgumentParser
import zstandard as zstd
import io
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import islice
from torch.nn.functional import sigmoid

# Define input and output paths
input_file = "data/en/1_sample.jsonl.zst"
output_file = "data/en/1_sample_registers.jsonl.zst"


# Data reading function for .zst files
def local_data(file_path):
    with open(file_path, "rb") as file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(file) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                yield line


# Processing function with batched encoding
def batch_process(model, tokenizer, input_path, batch_size=64, max_batch_length=12800):
    data_iterator = local_data(input_path)

    while True:
        # Step 1: Take a large chunk of data
        large_batch = list(islice(data_iterator, max_batch_length))
        if not large_batch:
            break  # End of data

        # Step 2: Extract and tokenize texts in a single pass
        text_data = []
        for idx, line in enumerate(large_batch):
            item = json.loads(line)
            text = item.get("text", "")
            if text:
                tokens = tokenizer(
                    text, truncation=True, max_length=512
                )  # Tokenize to dict, not tensor
                token_length = len(tokens["input_ids"])
                text_data.append(
                    (idx, item, tokens, token_length)
                )  # Store original index, item, tokens, and length

        # Step 3: Sort data by token length to minimize padding
        text_data.sort(key=lambda x: x[3])  # Sort by token_length

        # Step 4: Process sorted data in mini-batches of specified size
        processed_results = []
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i : i + batch_size]
            batch_indices = [item[0] for item in batch]
            batch_items = [item[1] for item in batch]
            batch_tokens = [item[2] for item in batch]  # Keep tokens as dictionaries

            # Use tokenizer's pad method to pad dynamically
            batch_tokens_padded = tokenizer.pad(
                batch_tokens, padding=True, return_tensors="pt"
            ).to(model.device)

            # Step 5: Run model inference
            with torch.no_grad():
                outputs = model(**batch_tokens_padded)
                probabilities = sigmoid(
                    outputs.logits
                )  # Apply sigmoid to get probabilities
                predicted_labels = (
                    probabilities > 0.5
                ).int()  # Binarize with threshold 0.5

            # Step 6: Add "registers" and "register_probabilities" to each original item and store results
        for idx, item_data, prob, pred_label in zip(
            batch_indices, batch_items, probabilities, predicted_labels
        ):
            item_data["registers"] = pred_label.nonzero(as_tuple=True)[0].tolist()
            item_data["register_probabilities"] = prob.tolist()
            processed_results.append((idx, item_data))

        # Step 7: Sort processed results back to original order
        processed_results.sort(key=lambda x: x[0])

        print(processed_results[0])
        exit()

        # Step 8: Yield results in the original order
        yield [item for _, item in processed_results]


# Example usage to process and write to output file
def process_and_save(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model.eval()
    # Get the original model's name and init tokenizer
    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    # Open output file with Zstandard compression
    with open(cfg.output_path, "wb") as out_file:
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(out_file) as writer:
            for processed_batch in batch_process(
                model, tokenizer, cfg.input_path, cfg.batch_size, cfg.max_batch_length
            ):
                for item in processed_batch:
                    # Write each item as a JSON line with the "register" field added
                    line = json.dumps(item) + "\n"
                    writer.write(line.encode("utf-8"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batch_length", type=int, default=12800)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_path", default="data/en/1_sample.jsonl.zst")
    parser.add_argument(
        "--output_path", default="data/en/1_sample_register_labels.jsonl.zst"
    )

    cfg = parser.parse_args()
    # Run the function to process the file and save the output
    process_and_save(cfg)

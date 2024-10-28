import os

os.environ["HF_HOME"] = ".hf/hf_home"

import json
import os
import io

from argparse import ArgumentParser

import requests
import torch
import zstandard as zstd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import sigmoid

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

# Flat list of labels
labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]

labels_upper = [x for x in labels_all if x.isupper()]


def decode_binary_labels(data, label_scheme_name):
    label_scheme = labels_all if label_scheme_name == "all" else labels_upper
    return [
        " ".join([label_scheme[i] for i, bin_val in enumerate(bin) if bin_val == 1])
        for bin in data
    ]


def stream_data(file_url):

    with requests.get(file_url, stream=True) as r:
        dctx = zstd.ZstdDecompressor()

        # Stream and decompress the data
        with dctx.stream_reader(r.raw) as reader:
            buffer = ""  # Initialize a buffer for partial lines
            for chunk in iter(lambda: reader.read(16384), b""):
                # Decode chunk and append to buffer, handle partial multi-byte characters
                buffer += chunk.decode("utf-8", errors="ignore")
                lines = buffer.split("\n")

                # Keep the last, potentially incomplete line in the buffer
                buffer = lines[-1]

                # Process complete lines
                for line in lines[:-1]:  # Exclude the last line because it's incomplete
                    if line:  # Ensure line is not empty
                        yield json.loads(line)


def local_data(data):
    for file_name in os.listdir(data):
        if file_name.endswith(".zst"):
            file_path = os.path.join(data, file_name)

            # Open the .zst file in binary mode
            with open(file_path, "rb") as file:
                # Initialize a streaming decompressor
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(file) as reader:
                    # Wrap the decompressed stream in a TextIOWrapper to read as text
                    text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_reader:
                        yield line


def process_batch(batch_texts, batch_docs, cfg, model, tokenizer, device):
    # Tokenize the batch
    inputs = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = sigmoid(outputs.logits)

    if cfg.train_labels == "all":
        # Ensure that subcategory has corresponding parent category
        for i in range(predictions.shape[0]):
            for (
                subcategory_index,
                parent_index,
            ) in subcategory_to_parent_index.items():
                if predictions[i, parent_index] < predictions[i, subcategory_index]:
                    predictions[i, parent_index] = predictions[i, subcategory_index]

    if cfg.train_labels == "all" and cfg.predict_labels == "upper":
        predictions = predictions[:, upper_all_indexes]

    for i, doc in enumerate(batch_docs):
        binary_predictions = predictions[i] > cfg.threshold
        predicted_labels = decode_binary_labels(
            [binary_predictions.tolist()], cfg.predict_labels
        )
        doc["registers"] = predicted_labels[0]

        # Write each document to file
        with open(cfg.output_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(doc, ensure_ascii=False) + "\n")


def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model.eval()

    # Get the original model's name and init tokenizer
    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    data_iterator = local_data(cfg.input_file)

    batch_texts = []
    batch_docs = []
    processed_batches = 0

    for doc in data_iterator:
        text = doc["text"]
        batch_texts.append(text)
        batch_docs.append(doc)

        # When the batch is full, process it
        if len(batch_texts) == cfg.batch_size:
            process_batch(batch_texts, batch_docs, cfg, model, tokenizer, device)
            batch_texts, batch_docs = [], []  # Reset for the next batch
            processed_batches += 1
            if cfg.n_batches and processed_batches >= cfg.n_batches:
                print("Done.")
                exit()

    # Don't forget to process the last batch if it's not empty
    if batch_texts:
        process_batch(batch_texts, batch_docs, cfg, model, tokenizer, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model_path", default="models/xlm-roberta-base")
    parser.add_argument("--input_file", default="data/en/1_sample.jsonl.zst")

    cfg = parser.parse_args()

    print(parser.dump(cfg))
    run(cfg)

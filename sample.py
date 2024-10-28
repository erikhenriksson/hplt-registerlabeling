import requests
import zstandard as zstd
import io

# Define the URLs and file paths
url = "https://data.hplt-project.org/two/cleaned/eng_Latn/1.jsonl.zst"
output_path = "data/en/1_sample.jsonl.zst"
target_lines = 10000

# Create zstd decompressor and compressor
dctx = zstd.ZstdDecompressor()
cctx = zstd.ZstdCompressor()

# Make the output directory if it doesn't exist
import os

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Open a streaming request and stream the zstd file
with requests.get(url, stream=True) as r, open(output_path, "wb") as f_out:
    # Ensure the request was successful
    r.raise_for_status()

    # Wrap the response in a streaming decompressor and start compressing the limited lines
    reader = dctx.stream_reader(r.raw)
    text_reader = io.TextIOWrapper(
        reader, encoding="utf-8"
    )  # Wrap in TextIOWrapper to read as text
    writer = cctx.stream_writer(f_out)

    # Read and write only the first 1000 lines
    lines = 0
    for line in text_reader:
        writer.write(line.encode("utf-8"))  # Compress as binary
        lines += 1
        if lines >= target_lines:
            break

    # Close the writer after writing desired lines
    writer.flush()
    writer.close()

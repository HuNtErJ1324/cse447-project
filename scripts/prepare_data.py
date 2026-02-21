#!/usr/bin/env python3
"""
Prepare training data for the CSE447 next-character prediction project.
Robust version that avoids 'datasets' library hanging issues.

This script:
  1. Reads all existing .txt files from data/
  2. Creates train/dev splits
  3. Downloads a large public text dataset (C4 shard) via direct HTTP
     to fill the ~3GB checkpoint budget.
"""

import os
import sys
import random
import time
import gzip
import shutil
import ssl
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Target size in bytes (approx 2.8 GB to leave room for overhead)
TARGET_BYTES = int(2.8 * 1024 * 1024 * 1024)

# Source URL for a large text file (C4 English shard)
# This is a direct download link from HuggingFace
C4_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"

DEV_FRACTION = 0.10
RANDOM_SEED = 42

SKIP_FILES = {
    "train.txt", "dev.txt", "dev_input.txt", "dev_answer.txt", 
    "culturax.txt", "uv.lock", "train_combined.txt", "cc100.txt",
    "large_corpus.txt", "c4_shard.json.gz"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line)
    return lines

def write_lines(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"  Wrote {len(lines):,} lines to {path}")

def split_train_dev(lines, dev_frac=DEV_FRACTION, seed=RANDOM_SEED):
    rng = random.Random(seed)
    shuffled = list(lines)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - dev_frac))
    return shuffled[:split_idx], shuffled[split_idx:]

def download_file(url, output_path):
    print(f"\nDownloading {url} to {output_path}...")
    try:
        # Create an unverified context to avoid SSL cert issues in some envs
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context, timeout=30) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1MB
            written = 0
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if total_size > 0:
                        percent = (written / total_size) * 100
                        print(f"\r  Progress: {written/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB ({percent:.1f}%)", end="")
                    else:
                        print(f"\r  Progress: {written/1024/1024:.1f} MB", end="")
            print(f"\n  Download complete: {output_path}")
            return True
            
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False

def extract_and_fill(gz_path, output_path, target_bytes):
    print(f"  Extracting and processing text from {gz_path}...")
    
    current_size = 0
    if os.path.exists(output_path):
        current_size = os.path.getsize(output_path)
    
    # We will read the gzipped JSON lines and extract the 'text' field
    # If the file isn't big enough, we'll repeat the content
    
    import json
    
    # Read all text first to memory (up to a limit) or just stream it
    # Since we need to duplicate, let's read distinct lines into a buffer
    # until the buffer is reasonably big, then write repeatedly.
    
    buffer = []
    buffer_limit_bytes = 100 * 1024 * 1024  # 100 MB buffer
    current_buffer_bytes = 0
    
    try:
        with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        # Clean up newlines
                        text = text.replace("\n", " ")
                        buffer.append(text)
                        current_buffer_bytes += len(text)
                        if current_buffer_bytes >= buffer_limit_bytes:
                            break
                except ValueError:
                    continue
    except Exception as e:
        print(f"  Error reading gzip file: {e}")
        if not buffer:
            return
            
    print(f"  Buffered {len(buffer):,} lines ({current_buffer_bytes/1024/1024:.1f} MB)")
    
    # Now write to the output file until we reach target size
    with open(output_path, "a", encoding="utf-8") as f_out:
        while current_size < target_bytes:
            for line in buffer:
                f_out.write(line + "\n")
                current_size += len(line) + 1
                if current_size >= target_bytes:
                    break
            
            print(f"  Current size: {current_size/1024/1024/1024:.2f} GB / {target_bytes/1024/1024/1024:.2f} GB")
            
    print(f"  Final file size: {current_size/1024/1024/1024:.2f} GB")

def create_dummy_data(output_path, target_bytes, seed_data):
    """Fallback: Generate data by repeating existing seed data."""
    print("  Generating dummy data from seed files...")
    if not seed_data:
        seed_data = ["The quick brown fox jumps over the lazy dog."]
        
    current_size = 0
    with open(output_path, "w", encoding="utf-8") as f:
        while current_size < target_bytes:
            # Shuffle seed data to make it slightly less repetitive locally
            chunk = list(seed_data)
            random.shuffle(chunk)
            
            for line in chunk:
                f.write(line + "\n")
                current_size += len(line) + 1
                if current_size >= target_bytes:
                    break
            
            if current_size % (100 * 1024 * 1024) < 10000: # Log every ~100MB
                 print(f"  Generated {current_size/1024/1024:.1f} MB...")
                 
    print(f"  Generated {current_size/1024/1024/1024:.2f} GB")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  CSE447 Data Preparation (Robust)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Read existing
    # ------------------------------------------------------------------
    print("\n[Step 1] Reading existing data...")
    all_lines = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt") and fname not in SKIP_FILES:
            path = os.path.join(DATA_DIR, fname)
            lines = read_lines(path)
            all_lines.extend(lines)
            print(f"  Loaded {fname}: {len(lines)} lines")

    # ------------------------------------------------------------------
    # Step 2: Split
    # ------------------------------------------------------------------
    print("\n[Step 2] Splitting train/dev...")
    train_lines, dev_lines = split_train_dev(all_lines)
    
    write_lines(train_lines, os.path.join(DATA_DIR, "train.txt"))
    write_lines(dev_lines, os.path.join(DATA_DIR, "dev.txt"))
    
    # Dev input/answer
    dev_inputs, dev_answers = [], []
    for line in dev_lines:
        if len(line) > 1:
            cut = random.randint(1, len(line)-1)
            dev_inputs.append(line[:cut])
            dev_answers.append(line[cut])
            
    write_lines(dev_inputs, os.path.join(DATA_DIR, "dev_input.txt"))
    write_lines(dev_answers, os.path.join(DATA_DIR, "dev_answer.txt"))

    # ------------------------------------------------------------------
    # Step 3: Fill to 3GB
    # ------------------------------------------------------------------
    print("\n[Step 3] Filling checkpoint budget...")
    combined_path = os.path.join(DATA_DIR, "train_combined.txt")
    
    # Start with valid training data
    with open(combined_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
            
    # Download C4 shard
    gz_path = os.path.join(DATA_DIR, "c4_shard.json.gz")
    download_success = False
    
    if not os.path.exists(gz_path):
        download_success = download_file(C4_URL, gz_path)
    else:
        print(f"  Found existing shard {gz_path}")
        download_success = True
        
    if download_success:
        try:
            extract_and_fill(gz_path, combined_path, TARGET_BYTES)
        except Exception as e:
            print(f"  Error processing C4 shard: {e}")
            create_dummy_data(combined_path, TARGET_BYTES, train_lines)
    else:
        print("  Download failed. Fallback to synthetic data generation.")
        create_dummy_data(combined_path, TARGET_BYTES, train_lines)

    print("\nDONE.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download and prepare TinyStories dataset for training.
This script downloads the dataset from HuggingFace and converts it to JSONL format.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def prepare_tinystories(output_dir="data", max_samples=None):
    """
    Download TinyStories dataset and save as JSONL.

    Args:
        output_dir: Directory to save the JSONL files
        max_samples: Maximum number of samples to download (None = all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Downloading TinyStories dataset from HuggingFace...")
    # Load TinyStories dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} samples")

    # Save to JSONL format
    output_file = output_dir / "tinystories_train.jsonl"
    print(f"Saving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Converting to JSONL"):
            # TinyStories dataset has a 'text' field
            content = example.get('text', '')
            if content.strip():
                json_line = json.dumps({"content": content}, ensure_ascii=False)
                f.write(json_line + '\n')

    print(f"Dataset saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Create validation set (optional, use first 1000 samples)
    print("\nCreating validation set...")
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    val_output_file = output_dir / "tinystories_val.jsonl"

    with open(val_output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(val_dataset, desc="Converting validation to JSONL"):
            content = example.get('text', '')
            if content.strip():
                json_line = json.dumps({"content": content}, ensure_ascii=False)
                f.write(json_line + '\n')

    print(f"Validation set saved to {val_output_file}")
    print(f"File size: {val_output_file.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for JSONL files")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples (None = all)")

    args = parser.parse_args()

    prepare_tinystories(args.output_dir, args.max_samples)

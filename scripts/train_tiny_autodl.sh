#!/bin/bash

# LLaDA Tiny Training Script for AutoDL RTX 5090 instances

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_ROOT}/outputs/tiny_test_autodl/logs"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${PROJECT_ROOT}" || exit 1

if [ ! -f "data/tinystories_train.jsonl" ]; then
    echo "Preparing TinyStories dataset..."
    python scripts/prepare_tinystories.py
fi

echo "Starting training on AutoDL..."
python -m accelerate.commands.launch \
    --config_file accelerate_configs/1_gpu.yaml \
    training/train_llada_tiny.py \
    config=configs/tiny_test_autodl.yaml

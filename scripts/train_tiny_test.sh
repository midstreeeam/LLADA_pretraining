#!/bin/bash

# LLaDA Tiny Test Training Script
# For RTX 4060 8GB GPU with TinyStories dataset

# Activate virtual environment
source /home/midstream/workspace/LLADA_pretraining/.venv/bin/activate
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR="outputs/tiny_test/logs"

# Change to the project directory
cd /home/midstream/workspace/LLADA_pretraining

# First, prepare the TinyStories dataset if not already done
if [ ! -f "data/tinystories_train.jsonl" ]; then
    echo "Preparing TinyStories dataset..."
    python scripts/prepare_tinystories.py
fi

# Run the training with accelerate (using tiny training script that initializes from scratch)
echo "Starting training..."
accelerate launch \
    --config_file accelerate_configs/1_gpu.yaml \
    training/train_llada_tiny.py \
    config=configs/tiny_test.yaml

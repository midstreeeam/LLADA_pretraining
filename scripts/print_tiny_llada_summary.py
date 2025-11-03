import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.configuration_llada import LLaDAConfig
from models.modeling_llada import LLaDAModelLM


def load_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    return cfg


def build_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_path, padding_side="left")

    mask_token = "<|mask|>"
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)

    llada_config_dict = {}
    if hasattr(cfg.model, "llada_config") and cfg.model.llada_config is not None:
        llada_config_dict = OmegaConf.to_container(cfg.model.llada_config, resolve=True)

    llada_config = LLaDAConfig(**llada_config_dict)
    llada_config.mask_token_id = mask_token_id

    model = LLaDAModelLM(config=llada_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.embedding_size = model.config.new_vocab_size
    model.config.mask_token_id = mask_token_id

    return model, tokenizer


def describe_model(model: LLaDAModelLM):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg = model.config

    print("=== LLaDA Tiny Summary ===")
    print(f"Hidden size (d_model): {cfg.d_model}")
    print(f"Layers (n_layers):     {cfg.n_layers}")
    print(f"Attention heads:       {cfg.n_heads}")
    print(f"KV heads:              {cfg.n_kv_heads}")
    print(f"MLP ratio:             {cfg.mlp_ratio}")
    print(f"Max sequence length:   {cfg.max_sequence_length}")
    print(f"Vocab size:            {cfg.vocab_size}")
    print(f"New vocab size:        {cfg.new_vocab_size}")
    print(f"Mask token id:         {cfg.mask_token_id}")
    print()
    print(f"Total parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")


def main():
    parser = argparse.ArgumentParser(description="Print a summary for the tiny LLaDA model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiny_test.yaml"),
        help="Path to the YAML config for the tiny model.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = build_model(cfg)
    describe_model(model)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

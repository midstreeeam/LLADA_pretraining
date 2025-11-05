#!/usr/bin/env python
"""
Generate text locally from a saved LLaDA checkpoint using the official sampling strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.configuration_llada import LLaDAConfig  # noqa: E402
from models.modeling_llada import LLaDAModelLM  # noqa: E402
from training.generate import diffusion_generate_text  # noqa: E402


def load_training_config(checkpoint: Path, config_path: Path | None) -> OmegaConf:
    if config_path is not None:
        return OmegaConf.load(config_path)

    default_cfg = checkpoint.parent / "config.yaml"
    if default_cfg.exists():
        return OmegaConf.load(default_cfg)

    raise FileNotFoundError(
        "Could not find training config. Pass --config pointing to the YAML file saved during training."
    )


def load_model(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> LLaDAModelLM:
    ckpt_dir = checkpoint / "unwrapped_model"
    config_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "pytorch_model.bin"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {ckpt_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing pytorch_model.bin in {ckpt_dir}")

    state_dict = torch.load(weights_path, map_location="cpu")
    num_embeddings = state_dict["model.transformer.wte.weight"].shape[0]

    config = LLaDAConfig.from_pretrained(str(ckpt_dir))
    config.embedding_size = num_embeddings
    config.vocab_size = num_embeddings

    model = LLaDAModelLM(config)
    state_dict.pop("model.transformer.ff_out.weight", None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] Unexpected keys ignored when loading state dict: {unexpected}")
    if missing:
        print(f"[warn] Missing keys when loading state dict: {missing}")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a LLaDA checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/tiny_test_autodl/checkpoint-630000"),
        help="Path to checkpoint directory (containing unwrapped_model/).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to the training YAML config. If omitted, tries checkpoint.parent/config.yaml.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate (official sampler uses all at once).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help="Number of diffusion steps for the official sampler.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 reproduces the official greedy + Gumbel behaviour).",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "Once upon a time, there was a little",
            "One day, a brave girl named",
            "In a magical forest, there lived",
            "A small boy found a toy and",
            "The happy cat wanted to",
        ],
        help="Prompts to condition on. Provide as space-separated strings or leave empty for defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")

    cfg = load_training_config(checkpoint, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = load_model(checkpoint, device, dtype)

    tokenizer_name = cfg.model.pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")

    # Ensure the <|mask|> token exists, mirroring training.
    mask_token = "<|mask|>"
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})

    max_seq_length = cfg.dataset.preprocessing.max_seq_length

    outputs = diffusion_generate_text(
        model=model,
        tokenizer=tokenizer,
        prompts=args.prompts,
        device=device,
        max_seq_length=max_seq_length,
        max_new_tokens=args.max_new_tokens,
        num_steps=args.num_steps,
        temperature=args.temperature,
        schedule="linear",
        strategy="official",
        block_length=None,
        remasking="low_confidence",
        cfg_scale=0.0,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        top_k=0,
        top_p=1.0,
        seed=42,
    )

    if isinstance(outputs, tuple):
        generated_texts, _ = outputs
    else:
        generated_texts: List[str] = outputs

    print("\n=== Generated Samples ===")
    for prompt, text in zip(args.prompts, generated_texts):
        print(f"\nPrompt: {prompt}\nOutput: {text}\n")


if __name__ == "__main__":
    main()

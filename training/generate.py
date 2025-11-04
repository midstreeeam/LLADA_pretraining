import logging
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from models.sampling import get_mask_schedule, top_k_top_p_filtering

logger = logging.getLogger(__name__)


@torch.no_grad()
def diffusion_generate_text(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: Optional[torch.device] = None,
    max_seq_length: int,
    max_new_tokens: int = 80,
    num_steps: int = 40,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    schedule: str = "cosine",
    seed: Optional[int] = None,
    return_debug: bool = False,
) -> List[str]:
    """
    Generate text with LLaDA using the iterative masking procedure that matches the
    diffusion-style training objective. The implementation follows the typical
    MaskGIT update scheme: start fully masked (except for the prompt), iteratively
    sample tokens, and progressively reduce the number of masked positions.
    """
    if not prompts:
        return []

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    # Make sure we can iterate the input more than once.
    prompts = [p if p is not None else "" for p in prompts]
    batch_size = len(prompts)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id

    eos_token_id = tokenizer.eos_token_id
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = eos_token_id

    if eos_token_id is None:
        raise ValueError("Tokenizer must define an eos_token_id for generation.")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    prompt_token_lists: List[List[int]] = []
    max_prompt_len = 0
    for prompt in prompts:
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not token_ids or token_ids[0] != bos_token_id:
            token_ids = [bos_token_id] + token_ids
        prompt_token_lists.append(token_ids)
        max_prompt_len = max(max_prompt_len, len(token_ids))

    seq_len = min(max_seq_length, max_prompt_len + max_new_tokens + 1)
    if seq_len < 2:
        raise ValueError("Effective sequence length is too small for generation.")

    prompt_template = torch.full(
        (batch_size, seq_len), fill_value=mask_token_id, dtype=torch.long, device=device
    )
    fixed_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    token_steps = torch.full((batch_size, seq_len), fill_value=-2, dtype=torch.int32, device=device)

    for idx, tokens in enumerate(prompt_token_lists):
        # Reserve the last token for EOS so the diffusion model keeps the same framing as training.
        usable_prompt_len = min(len(tokens), seq_len - 1)
        if usable_prompt_len > 0:
            prompt_template[idx, :usable_prompt_len] = torch.tensor(
                tokens[:usable_prompt_len], dtype=torch.long, device=device
            )
            fixed_mask[idx, :usable_prompt_len] = True
        prompt_template[idx, seq_len - 1] = eos_token_id
        fixed_mask[idx, seq_len - 1] = True
        prompt_token_lists[idx] = tokens[:usable_prompt_len]
        token_steps[idx, :usable_prompt_len] = -1
        token_steps[idx, seq_len - 1] = -1

    input_ids = prompt_template.clone()
    input_ids[~fixed_mask] = mask_token_id
    remain_mask = ~fixed_mask
    total_maskable = remain_mask.sum(dim=1)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    was_training = model.training
    model.eval()

    mask_schedule = get_mask_schedule(schedule)
    for step in range(num_steps):
        if remain_mask.sum() == 0:
            break

        model_inputs = input_ids.clone()
        model_inputs[remain_mask] = mask_token_id

        attention_bias = torch.ones(
            input_ids.size(0),
            1,
            input_ids.size(1),
            input_ids.size(1),
            device=device,
            dtype=torch.float32,
        )

        logits = model(model_inputs, attention_bias=attention_bias).logits
        logits = logits.to(torch.float32)

        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        logits = logits / temperature

        if top_k > 0 or top_p < 1.0:
            logits_view = logits.view(-1, logits.size(-1))
            logits_view = top_k_top_p_filtering(
                logits_view, top_k=max(top_k, 0), top_p=top_p, min_tokens_to_keep=1
            )
            logits = logits_view.view_as(logits)

        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(min=1e-9)

        sampled = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1,
        ).view(batch_size, seq_len)

        input_ids = torch.where(remain_mask, sampled, input_ids)
        input_ids = torch.where(fixed_mask, prompt_template, input_ids)
        token_steps = torch.where(fixed_mask, torch.full_like(token_steps, -1), token_steps)

        confidences = probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        confidences = torch.where(fixed_mask, torch.ones_like(confidences), confidences)

        if step == num_steps - 1:
            token_steps = torch.where(remain_mask, torch.full_like(token_steps, step + 1), token_steps)
            remain_mask.zero_()
            break

        progress = torch.tensor((step + 1) / num_steps, device=device)
        mask_ratio = mask_schedule(progress).clamp(min=0.0, max=1.0)

        desired_mask = (mask_ratio * total_maskable.float()).round().long()
        current_mask = remain_mask.sum(dim=1)
        desired_mask = torch.minimum(desired_mask, current_mask)

        maskable_mask = ~fixed_mask
        masked_conf = confidences.masked_fill(~maskable_mask, float("inf"))
        sorted_idx = torch.argsort(masked_conf, dim=1, descending=False)

        new_remain_mask = torch.zeros_like(remain_mask)
        for idx in range(batch_size):
            k = int(desired_mask[idx].item())
            if k <= 0:
                continue
            candidates = sorted_idx[idx]
            # Drop indices that map to fixed locations (infinite confidence).
            candidates = candidates[maskable_mask[idx, candidates]]
            if candidates.numel() == 0:
                continue
            k = min(k, candidates.numel())
            selected = candidates[:k]
            new_remain_mask[idx, selected] = True

        newly_filled = remain_mask & ~new_remain_mask
        if newly_filled.any():
            fill_steps = torch.full_like(token_steps, step + 1)
            token_steps = torch.where(newly_filled, fill_steps, token_steps)
        remain_mask = new_remain_mask

    if was_training:
        model.train()

    generated_texts: List[str] = []
    debug_annotations: Optional[List[List[str]]] = [] if return_debug else None
    for idx in range(batch_size):
        tokens = input_ids[idx].tolist()
        # Remove the leading BOS token (if present) to avoid duplicated <|endoftext|>.
        if bos_token_id is not None and tokens and tokens[0] == bos_token_id:
            tokens = tokens[1:]

        # Stop at the first EOS token.
        if eos_token_id in tokens:
            eos_pos = tokens.index(eos_token_id)
            tokens = tokens[:eos_pos]

        if pad_token_id is not None:
            tokens = [tid for tid in tokens if tid != pad_token_id]

        # Drop residual mask tokens.
        tokens = [tid for tid in tokens if tid != mask_token_id]

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        generated_texts.append(text.strip())

        if return_debug:
            full_tokens = input_ids[idx].tolist()
            steps = token_steps[idx].tolist()
            token_strs = tokenizer.convert_ids_to_tokens(full_tokens)
            annotated = []
            for tok, step_idx in zip(token_strs, steps):
                if step_idx == -1:
                    tag = "P"
                elif step_idx == -2:
                    tag = "?"
                else:
                    tag = str(step_idx)
                annotated.append(f"{tok}[{tag}]")
            debug_annotations.append(annotated)

    if return_debug:
        return generated_texts, debug_annotations  # type: ignore[return-value]
    return generated_texts

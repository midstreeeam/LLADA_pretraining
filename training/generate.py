import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.sampling import get_mask_schedule, top_k_top_p_filtering

logger = logging.getLogger(__name__)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Add Gumbel-style perturbations (official LLaDA decoding).
    """
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64).clamp_min(1e-12)
    gumbel_noise = (-torch.log(noise)) ** temperature
    perturbed = logits64.exp() / gumbel_noise
    return perturbed.to(logits.dtype)


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Compute the number of tokens to finalize per step (official sampler).
    """
    if steps <= 0:
        steps = 1
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    num_transfer_tokens += base
    for i in range(mask_num.size(0)):
        if steps > 0 and remainder[i].item() > 0:
            num_transfer_tokens[i, : remainder[i].item()] += 1
    return num_transfer_tokens


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
    strategy: str = "ratio",
    selection_scope: str = "all",
    tokens_per_step: Optional[int] = None,
    block_length: Optional[int] = None,
    remasking: str = "low_confidence",
    cfg_scale: float = 0.0,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    mask_token_override: Optional[int] = None,
    forbid_logits_token_ids: Optional[Iterable[int]] = None,
    forbid_confidence_token_ids: Optional[Iterable[int]] = None,
) -> List[str]:
    """
    Generate text with LLaDA using one of three strategies:
      * ratio: progressive masking ratio (MaskGIT-style)
      * fixed_tokens: finalize a fixed number of tokens per step
      * official: replicate the official LLaDA sampling algorithm (block diffusion)
    """
    if not prompts:
        return []

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

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
    if mask_token_override is not None:
        mask_token_id = int(mask_token_override)
    if mask_token_id is None:
        mask_token_id = eos_token_id

    if eos_token_id is None:
        raise ValueError("Tokenizer must define an eos_token_id for generation.")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    strategy = strategy.lower()
    selection_scope = selection_scope.lower()
    if strategy not in {"ratio", "fixed_tokens", "official"}:
        raise ValueError(f"Unknown generation strategy: {strategy}")
    if selection_scope not in {"all", "masked_only"}:
        raise ValueError(f"Unknown selection_scope: {selection_scope}")
    if tokens_per_step is not None and tokens_per_step < 0:
        raise ValueError("tokens_per_step must be non-negative if provided.")

    forbid_logits_ids = None if forbid_logits_token_ids is None else [int(x) for x in forbid_logits_token_ids]
    forbid_confidence_ids = None if forbid_confidence_token_ids is None else [int(x) for x in forbid_confidence_token_ids]
    if logits_eos_inf and forbid_logits_ids is None:
        forbid_logits_ids = [126081]
    if confidence_eos_eot_inf and forbid_confidence_ids is None:
        forbid_confidence_ids = [126081, 126348]

    prompt_token_lists: List[List[int]] = []
    max_prompt_len = 0
    for prompt in prompts:
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not token_ids or token_ids[0] != bos_token_id:
            token_ids = [bos_token_id] + token_ids
        prompt_token_lists.append(token_ids)
        max_prompt_len = max(max_prompt_len, len(token_ids))

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    if strategy == "official":
        generated_texts, debug_annotations = _official_generate(
            model=model,
            tokenizer=tokenizer,
            prompt_token_lists=prompt_token_lists,
            device=device,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
            temperature=temperature,
            block_length=block_length,
            remasking=remasking,
            cfg_scale=cfg_scale,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            return_debug=return_debug,
            forbid_logits_ids=forbid_logits_ids,
            forbid_confidence_ids=forbid_confidence_ids,
        )
        if return_debug:
            return generated_texts, debug_annotations  # type: ignore[return-value]
        return generated_texts

    seq_len = min(max_seq_length, max_prompt_len + max_new_tokens + 1)
    if seq_len < 2:
        raise ValueError("Effective sequence length is too small for generation.")

    prompt_template = torch.full(
        (batch_size, seq_len), fill_value=mask_token_id, dtype=torch.long, device=device
    )
    fixed_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    token_steps = torch.full((batch_size, seq_len), fill_value=-2, dtype=torch.int32, device=device)

    for idx, tokens in enumerate(prompt_token_lists):
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

    was_training = model.training
    model.eval()

    mask_schedule = get_mask_schedule(schedule)

    finalize_plan: Optional[torch.Tensor] = None
    if strategy == "fixed_tokens":
        plan = torch.zeros(batch_size, num_steps, dtype=torch.long, device=device)
        remaining_initial = total_maskable.clone()
        if tokens_per_step is not None and num_steps > 0:
            plan.fill_(int(tokens_per_step))
            planned = plan[:, :-1].sum(dim=1) if num_steps > 1 else torch.zeros(batch_size, device=device, dtype=torch.long)
            plan[:, -1] = torch.clamp(remaining_initial - planned, min=0)
        elif num_steps > 0:
            base = torch.div(remaining_initial, num_steps, rounding_mode="floor")
            remainder = remaining_initial % num_steps
            plan += base.unsqueeze(1)
            if num_steps > 0:
                step_indices = torch.arange(num_steps, device=device)
                plan += (step_indices.unsqueeze(0) < remainder.unsqueeze(1)).long()
        finalize_plan = plan

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

        probs = F.softmax(logits, dim=-1).clamp(min=1e-9)

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
            fill_steps = torch.full_like(token_steps, step + 1)
            token_steps = torch.where(remain_mask, fill_steps, token_steps)
            remain_mask.zero_()
            break

        if strategy == "ratio":
            progress = torch.tensor((step + 1) / num_steps, device=device)
            mask_ratio = mask_schedule(progress).clamp(min=0.0, max=1.0)

            desired_mask = (mask_ratio * total_maskable.float()).round().long()
            current_mask = remain_mask.sum(dim=1)
            desired_mask = torch.minimum(desired_mask, current_mask)

            maskable_mask = (~fixed_mask) if selection_scope == "all" else remain_mask.clone()
            masked_conf = confidences.masked_fill(~maskable_mask, float("inf"))
            sorted_idx = torch.argsort(masked_conf, dim=1, descending=False)

            new_remain_mask = torch.zeros_like(remain_mask)
            for idx in range(batch_size):
                k = int(desired_mask[idx].item())
                if k <= 0:
                    continue
                candidates = sorted_idx[idx]
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
        else:
            assert finalize_plan is not None
            step_counts = finalize_plan[:, step]
            remaining_now = remain_mask.sum(dim=1)
            steps_left = num_steps - step
            step_counts = torch.minimum(step_counts, remaining_now)
            need_fill = (remaining_now > 0) & (step_counts == 0)
            if steps_left > 1:
                step_counts = torch.where(need_fill, torch.ones_like(step_counts), step_counts)
                step_counts = torch.minimum(step_counts, remaining_now)

            confidences_masked = confidences.masked_fill(~remain_mask, float("-inf"))
            newly_filled = torch.zeros_like(remain_mask)
            for idx in range(batch_size):
                k = int(step_counts[idx].item())
                if k <= 0:
                    continue
                available = torch.nonzero(remain_mask[idx], as_tuple=False).view(-1)
                if available.numel() == 0:
                    continue
                k = min(k, available.numel())
                _, indices = torch.topk(confidences_masked[idx], k=k, largest=True)
                newly_filled[idx, indices] = True

            if newly_filled.any():
                fill_steps = torch.full_like(token_steps, step + 1)
                token_steps = torch.where(newly_filled, fill_steps, token_steps)
                remain_mask = remain_mask & ~newly_filled

    if was_training:
        model.train()

    generated_texts: List[str] = []
    debug_annotations: Optional[List[List[str]]] = [] if return_debug else None
    for idx in range(batch_size):
        tokens = input_ids[idx].tolist()
        if bos_token_id is not None and tokens and tokens[0] == bos_token_id:
            tokens = tokens[1:]

        if eos_token_id in tokens:
            eos_pos = tokens.index(eos_token_id)
            tokens = tokens[:eos_pos]

        if pad_token_id is not None:
            tokens = [tid for tid in tokens if tid != pad_token_id]
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


def _official_generate(
    *,
    model,
    tokenizer,
    prompt_token_lists: List[List[int]],
    device: torch.device,
    max_seq_length: int,
    max_new_tokens: int,
    num_steps: int,
    temperature: float,
    block_length: Optional[int],
    remasking: str,
    cfg_scale: float,
    mask_token_id: int,
    pad_token_id: Optional[int],
    eos_token_id: Optional[int],
    bos_token_id: Optional[int],
    return_debug: bool,
    forbid_logits_ids: Optional[List[int]],
    forbid_confidence_ids: Optional[List[int]],
) -> Tuple[List[str], Optional[List[List[str]]]]:
    results: List[str] = []
    debug_annotations: Optional[List[List[str]]] = [] if return_debug else None

    for tokens in prompt_token_lists:
        tokens = tokens[: max_seq_length]
        prompt_len = len(tokens)
        available = max(0, max_seq_length - prompt_len)
        gen_len = min(max_new_tokens, available)
        if gen_len <= 0:
            decode_tokens = tokens[1:] if bos_token_id is not None and tokens and tokens[0] == bos_token_id else tokens
            text = tokenizer.decode(decode_tokens, skip_special_tokens=True).strip()
            results.append(text)
            if return_debug:
                steps = [-1 if idx < prompt_len else -2 for idx in range(len(tokens))]
                token_strs = tokenizer.convert_ids_to_tokens(tokens)
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
            continue

        seq_len = prompt_len + gen_len
        x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(tokens, dtype=torch.long, device=device)
        prompt_index = (x != mask_token_id)

        token_steps = torch.full((seq_len,), -2, dtype=torch.int32, device=device)
        token_steps[:prompt_len] = -1

        if block_length is None or block_length <= 0 or block_length > gen_len:
            block_length_eff = gen_len
        else:
            block_length_eff = block_length

        if gen_len % block_length_eff != 0:
            raise ValueError("For official generation, block_length must divide the generation length.")

        num_blocks = gen_len // block_length_eff
        if num_blocks <= 0:
            num_blocks = 1
        if num_steps % num_blocks != 0:
            raise ValueError("For official generation, num_steps must be divisible by (gen_len / block_length).")
        steps_per_block = num_steps // num_blocks

        remain_blocks = gen_len
        current_offset = prompt_len
        global_step = 0

        for block_idx in range(num_blocks):
            current_block_len = min(block_length_eff, remain_blocks)
            block_start = current_offset
            block_end = current_offset + current_block_len
            remain_blocks -= current_block_len
            current_offset += current_block_len

            block_mask_index = (x[:, block_start:block_end] == mask_token_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for inner_step in range(steps_per_block):
                global_step += 1
                mask_index = (x == mask_token_id)
                if mask_index.sum() == 0:
                    break

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_token_id
                    inputs = torch.cat([x, un_x], dim=0)
                    attention_bias = torch.ones(
                        inputs.size(0), 1, inputs.size(1), inputs.size(1), dtype=torch.float32, device=device
                    )
                    logits = model(inputs, attention_bias=attention_bias).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
                else:
                    attention_bias = torch.ones(
                        x.size(0), 1, x.size(1), x.size(1), dtype=torch.float32, device=device
                    )
                    logits = model(x, attention_bias=attention_bias).logits

                if forbid_logits_ids:
                    for token_id in forbid_logits_ids:
                        logits[:, :, token_id] = float("-inf")

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if forbid_confidence_ids:
                    for token_id in forbid_confidence_ids:
                        logits_with_noise[:, :, token_id] = float("-inf")

                if remasking == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")

                block_range_mask = torch.zeros_like(mask_index)
                block_range_mask[:, block_start:block_end] = True
                x0_p = torch.where(block_range_mask, x0_p, torch.full_like(x0_p, float("-inf")))

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

                k = int(num_transfer_tokens[0, inner_step].item())
                available = torch.nonzero(mask_index & block_range_mask, as_tuple=False).flatten()
                if available.numel() == 0 or k <= 0:
                    continue
                k = min(k, available.numel())
                confidence_values = confidence[0, available]
                _, select_index = torch.topk(confidence_values, k=k, largest=True)
                selected_positions = available[select_index]

                x[0, selected_positions] = x0[0, selected_positions]
                token_steps[selected_positions] = global_step

        final_tokens = x[0].tolist()
        if bos_token_id is not None and final_tokens and final_tokens[0] == bos_token_id:
            decode_tokens = final_tokens[1:]
            decode_steps = token_steps[1:].tolist()
        else:
            decode_tokens = final_tokens
            decode_steps = token_steps.tolist()

        if eos_token_id is not None and eos_token_id in decode_tokens:
            eos_pos = decode_tokens.index(eos_token_id)
            decode_tokens = decode_tokens[:eos_pos]
            decode_steps = decode_steps[:eos_pos]

        if pad_token_id is not None:
            filtered = [(tid, step) for tid, step in zip(decode_tokens, decode_steps) if tid != pad_token_id]
            if filtered:
                decode_tokens, decode_steps = zip(*filtered)
                decode_tokens = list(decode_tokens)
                decode_steps = list(decode_steps)
            else:
                decode_tokens, decode_steps = [], []

        decode_tokens = [tid for tid in decode_tokens if tid != mask_token_id]
        text = tokenizer.decode(decode_tokens, skip_special_tokens=True).strip()
        results.append(text)

        if return_debug:
            token_strs = tokenizer.convert_ids_to_tokens(final_tokens)
            all_steps = token_steps.tolist()
            annotated = []
            for tok, step_idx in zip(token_strs, all_steps):
                if step_idx == -1:
                    tag = "P"
                elif step_idx == -2:
                    tag = "?"
                else:
                    tag = str(step_idx)
                annotated.append(f"{tok}[{tag}]")
            debug_annotations.append(annotated)

    return results, debug_annotations

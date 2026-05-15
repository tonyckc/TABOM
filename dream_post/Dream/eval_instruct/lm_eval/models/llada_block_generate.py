"""
LLaDA block diffusion decode (aligned with CCD_lm_eval_v0 diffllm.generate and build_single_trajectory_llada).

Shared by lm_eval.models.diffllm and trajectory scripts; torch / torch.nn.functional only.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def normalize_llada_alg(alg: str) -> str:
    if alg in (None, ""):
        return "default"
    a = str(alg).strip().lower()
    if a in ("llada_original", "llada_ours"):
        return "default"
    if a in ("topk_margin", "entropy", "default", "credit"):
        return a
    raise ValueError(
        f"Unknown LLaDA alg={alg!r}; expected llada_original|llada_ours|topk_margin|entropy|default|credit"
    )


def apply_credit_fusion(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    state: Optional[Tuple[torch.Tensor, int]],
    *,
    credit_alpha: float = 0.7,
    boost_gamma: float = 0.2,
    decay_beta: float = 0.8,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    EMA-based credit fusion (matches eval diffllm._apply_credit_fusion):
    Maintain (B, L, V) credit, accumulate boosted top-1 prob on mask positions only, then
    ``fused_logits = logits + credit_alpha * log(mat + 1)``.
    Reset ``state=None`` outside each decode block.
    """
    B, L, V = logits.shape
    device = logits.device
    dtype = logits.dtype

    if state is None:
        mat = torch.zeros((B, L, V), dtype=dtype, device=device)
        iter_idx = 0
    else:
        mat, iter_idx = state
        if mat.shape != (B, L, V) or mat.device != device or mat.dtype != dtype:
            mat = torch.zeros((B, L, V), dtype=dtype, device=device)
            iter_idx = 0

    if iter_idx > 0:
        mat.mul_(decay_beta)

    probs = F.softmax(logits, dim=-1)
    top1_probs, top1_idx = torch.max(probs, dim=-1)

    enhanced = top1_probs.pow(boost_gamma).to(mat.dtype)
    update_vals = enhanced * mask_index.to(enhanced.dtype)
    mat.scatter_add_(2, top1_idx.unsqueeze(-1), update_vals.unsqueeze(-1))

    fused_logits = logits + credit_alpha * torch.log(mat + 1)
    new_state: Tuple[torch.Tensor, int] = (mat, iter_idx + 1)
    return fused_logits, new_state


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def resolve_mask_id(tokenizer, model: torch.nn.Module) -> int:
    mid = getattr(tokenizer, "mask_token_id", None)
    if mid is not None:
        return int(mid)
    cfg = getattr(model, "config", None)
    mid = getattr(cfg, "mask_token_id", None) if cfg is not None else None
    if mid is not None:
        return int(mid)
    raise ValueError("Cannot resolve mask_token_id from tokenizer/model.config")


def first_eos_response_meta(
    x: torch.Tensor,
    decode_step_map: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    eos_token_id: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Per trajectory: in response band ``[prompt_len, prompt_len+gen_length)``, find the **first**
    token equal to ``eos_token_id`` in generation order; return its buffer index and unmask step from
    ``decode_step_map`` (same ``global_step`` as ``entropy_trace[*]['step']``).

    If no EOS or no ``eos_token_id``, fields are ``None``.
    ``decode_step_map`` still ``-1`` on decoded EOS (should not happen) is recorded as ``None``.
    """
    out: List[Dict[str, Any]] = []
    if eos_token_id is None:
        for _ in range(int(x.shape[0])):
            out.append(
                {
                    "first_eos_resp_index": None,
                    "first_eos_decode_step": None,
                    "eos_token_id": None,
                }
            )
        return out

    eos = int(eos_token_id)
    bsz = int(x.shape[0])
    xcpu = x.detach().cpu()
    dsm = decode_step_map.detach().cpu()

    for b in range(bsz):
        first_idx: Optional[int] = None
        first_step: Optional[int] = None
        for k in range(gen_length):
            tok = int(xcpu[b, prompt_len + k].item())
            if tok != eos:
                continue
            first_idx = k
            st = int(dsm[b, k].item())
            first_step = st if st >= 0 else None
            break
        out.append(
            {
                "first_eos_resp_index": first_idx,
                "first_eos_decode_step": first_step,
                "eos_token_id": eos,
            }
        )
    return out


def _leftmost_eos_resp_index(
    x: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    eos_token_id: Optional[int],
    batch_row: int = 0,
) -> Optional[int]:
    """Leftmost ``eos`` index in ``[prompt_len, prompt_len+gen_length)``; ``None`` if none."""
    if eos_token_id is None:
        return None
    eos = int(eos_token_id)
    br = int(batch_row)
    for k in range(gen_length):
        if int(x[br, prompt_len + k].item()) == eos:
            return int(k)
    return None


def summarize_first_eos_samples(
    per_sample: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Simple aggregate over a batch list from ``first_eos_response_meta`` (for summary.json)."""
    steps = [int(r["first_eos_decode_step"]) for r in per_sample if r.get("first_eos_decode_step") is not None]
    idxs = [int(r["first_eos_resp_index"]) for r in per_sample if r.get("first_eos_resp_index") is not None]
    summary: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "n_samples": len(per_sample),
        "n_with_eos_in_response": len(steps),
    }
    if steps:
        summary["mean_first_eos_decode_step"] = float(statistics.mean(steps))
        summary["median_first_eos_decode_step"] = float(statistics.median(steps))
    if idxs:
        summary["mean_first_eos_resp_index"] = float(statistics.mean(idxs))
        summary["median_first_eos_resp_index"] = float(statistics.median(idxs))
    return summary


def _mean_shannon_entropy_llada_resp_masked(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    eps: float = 1e-10,
) -> Tuple[Optional[float], int]:
    """
    On still-[MASK] response positions, softmax raw logits to p, then mean Shannon entropy
    H = -sum p log p (matches Dream trajectory stats; no temperature / top-p).
    """
    B, L, _ = logits.shape
    device = logits.device
    pos = torch.arange(L, device=device)
    resp_band = (pos >= prompt_len) & (pos < prompt_len + gen_length)
    rows_list = []
    for b in range(B):
        sel = mask_index[b] & resp_band
        if sel.any():
            rows_list.append(logits[b, sel, :].float())
    if not rows_list:
        return None, 0
    sub = torch.cat(rows_list, dim=0)
    p = F.softmax(sub, dim=-1)
    H = -(p * torch.log(p + eps)).sum(dim=-1)
    return float(H.mean().item()), int(H.numel())


def _mean_max_prob_llada_resp_masked(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    prompt_len: int,
    gen_length: int,
) -> Tuple[Optional[float], int]:
    """
    On still-[MASK] response positions, softmax raw logits and take per-row max prob (top-1),
    then arithmetic mean over all such positions.
    """
    B, L, _ = logits.shape
    device = logits.device
    pos = torch.arange(L, device=device)
    resp_band = (pos >= prompt_len) & (pos < prompt_len + gen_length)
    rows_list = []
    for b in range(B):
        sel = mask_index[b] & resp_band
        if sel.any():
            rows_list.append(logits[b, sel, :].float())
    if not rows_list:
        return None, 0
    sub = torch.cat(rows_list, dim=0)
    p = F.softmax(sub, dim=-1)
    mx, _ = p.max(dim=-1)
    return float(mx.mean().item()), int(mx.numel())


def _var_shannon_entropy_llada_resp_masked(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    eps: float = 1e-10,
    resp_rel_end_exclusive: Optional[int] = None,
) -> Tuple[Optional[float], int]:
    """
    Per batch row with [MASK] in response: population variance of Shannon H over mask positions,
    then mean over rows with >=2 masks. Matches Dream ``_var_shannon_entropy_resp_masked``.

    ``resp_rel_end_exclusive``: if ``K``, only mask with gen index ``j < K`` (before first EOS);
    ``None`` means ``j < gen_length`` (full response band).
    """
    B, L, _ = logits.shape
    device = logits.device
    pos = torch.arange(L, device=device)
    rel_end = gen_length if resp_rel_end_exclusive is None else min(gen_length, max(0, int(resp_rel_end_exclusive)))
    resp_band = (pos >= prompt_len) & (pos < prompt_len + rel_end)
    row_vars: List[torch.Tensor] = []
    for b in range(B):
        sel = mask_index[b] & resp_band
        if int(sel.sum().item()) < 2:
            continue
        sub = logits[b, sel, :].float()
        p = F.softmax(sub, dim=-1)
        H = -(p * torch.log(p + eps)).sum(dim=-1)
        row_vars.append(H.var(unbiased=False))
    if not row_vars:
        return None, 0
    stacked = torch.stack(row_vars)
    return float(stacked.mean().item()), len(row_vars)


def _var_max_prob_llada_resp_masked(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    resp_rel_end_exclusive: Optional[int] = None,
) -> Tuple[Optional[float], int]:
    """Same as entropy variance but on softmax top-1 prob at mask positions. ``resp_rel_end_exclusive`` as above."""
    B, L, _ = logits.shape
    device = logits.device
    pos = torch.arange(L, device=device)
    rel_end = gen_length if resp_rel_end_exclusive is None else min(gen_length, max(0, int(resp_rel_end_exclusive)))
    resp_band = (pos >= prompt_len) & (pos < prompt_len + rel_end)
    row_vars: List[torch.Tensor] = []
    for b in range(B):
        sel = mask_index[b] & resp_band
        if int(sel.sum().item()) < 2:
            continue
        sub = logits[b, sel, :].float()
        p = F.softmax(sub, dim=-1)
        mx, _ = p.max(dim=-1)
        row_vars.append(mx.var(unbiased=False))
    if not row_vars:
        return None, 0
    stacked = torch.stack(row_vars)
    return float(stacked.mean().item()), len(row_vars)


def _pmax_decode_residual_llada_resp(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    transfer_index: torch.Tensor,
    prompt_len: int,
    gen_length: int,
) -> Tuple[Optional[float], int, int]:
    """
    This step: raw-logits softmax top-1 prob ``pmax`` at each still-[MASK] position.

    - **Decoded**: ``transfer_index & mask_index`` in response ``[prompt_len, prompt_len+gen_length)``;
      **max** of their ``pmax`` (per row, then mean over valid rows).
    - **Other masked**: same band, masked but not transferred this step; **mean** of ``pmax``.
    - **Residual** = (decoded max) - (other mean), per row then averaged; skip rows missing either;
      all skipped -> (None, 0, 0).

    Independent of ``alg`` confidence (margin / neg-entropy); always softmax top-1 prob.
    """
    B, L, _ = logits.shape
    device = logits.device
    pos = torch.arange(L, device=device)
    resp = ((pos >= prompt_len) & (pos < prompt_len + gen_length)).unsqueeze(0).expand(B, -1)
    dec = transfer_index & mask_index & resp
    oth = mask_index & (~transfer_index) & resp

    p = F.softmax(logits.float(), dim=-1)
    top1, _ = p.max(dim=-1)

    row_vals: List[torch.Tensor] = []
    n_dec_total = 0
    n_oth_total = 0
    for b in range(B):
        db, ob = dec[b], oth[b]
        if not db.any() or not ob.any():
            continue
        n_dec_total += int(db.sum().item())
        n_oth_total += int(ob.sum().item())
        r = top1[b, db].max() - top1[b, ob].mean()
        row_vals.append(r)
    if not row_vals:
        return None, 0, 0
    stacked = torch.stack(row_vals)
    return float(stacked.mean().item()), n_dec_total, n_oth_total


def _entropy_decode_residual_llada_resp(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    transfer_index: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    eps: float = 1e-10,
) -> Tuple[Optional[float], int, int]:
    """
    Matches Dream ``build_single_trajectory._entropy_decode_residual_dream_resp``:
    Shannon H per position from raw-logits softmax; ``mean(H_dec)`` on transferred response masks,
    ``mean(H_oth)`` on other masked response positions; return ``H_oth - H_dec`` (often positive
    when decoded positions are sharper). Per-row then mean over valid rows.
    """
    B, L, _ = logits.shape
    device = logits.device
    pos = torch.arange(L, device=device)
    resp = ((pos >= prompt_len) & (pos < prompt_len + gen_length)).unsqueeze(0).expand(B, -1)
    dec = transfer_index & mask_index & resp
    oth = mask_index & (~transfer_index) & resp

    p = F.softmax(logits.float(), dim=-1)
    H = -(p * torch.log(p + eps)).sum(dim=-1)

    row_vals: List[torch.Tensor] = []
    n_dec_total = 0
    n_oth_total = 0
    for b in range(B):
        db, ob = dec[b], oth[b]
        if not db.any() or not ob.any():
            continue
        n_dec_total += int(db.sum().item())
        n_oth_total += int(ob.sum().item())
        r = H[b, ob].mean() - H[b, db].mean()
        row_vals.append(r)
    if not row_vals:
        return None, 0, 0
    stacked = torch.stack(row_vals)
    return float(stacked.mean().item()), n_dec_total, n_oth_total


def normalize_trace_stat(s: str) -> str:
    t = (s or "entropy").strip().lower()
    if t in ("entropy", "h", "shannon"):
        return "entropy"
    if t in ("maxprob", "max_prob", "top1", "pmax", "max-p"):
        return "maxprob"
    if t in (
        "pmax_residual",
        "pmax_decode_residual",
        "decode_pmax_residual",
        "residual",
        "top1_residual",
    ):
        return "pmax_residual"
    if t in (
        "decode_entropy_residual",
        "entropy_decode_residual",
        "entropy_residual",
    ):
        return "decode_entropy_residual"
    if t in (
        "both_residuals",
        "both_decode_residuals",
        "decode_residuals_both",
        "entropy_and_pmax_residual",
    ):
        return "both_residuals"
    if t in (
        "masked_entropy_var",
        "var_masked_entropy",
        "entropy_var",
        "var_entropy_masked",
    ):
        return "masked_entropy_var"
    if t in (
        "masked_maxprob_var",
        "var_masked_maxprob",
        "maxprob_var",
        "var_maxprob_masked",
        "top1_var",
    ):
        return "masked_maxprob_var"
    if t in (
        "both_masked_vars",
        "masked_vars_both",
        "var_both",
    ):
        return "both_masked_vars"
    raise ValueError(
        f"Unknown trace_stat={s!r}; use 'entropy' | 'maxprob' | 'pmax_residual' | "
        f"'decode_entropy_residual' | 'both_residuals' | 'masked_entropy_var' | "
        f"'masked_maxprob_var' | 'both_masked_vars'"
    )


@torch.no_grad()
def llada_generate_with_tracking(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    mask_id: int,
    alg: str,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: Optional[int] = None,
    credit_alpha: float = 0.7,
    credit_boost_gamma: float = 0.2,
    credit_decay_beta: float = 0.8,
    confidence_threshold: Optional[float] = None,
    fixed_confidence_topk: Optional[int] = None,
    return_entropy_trace: bool = False,
    trace_stat: str = "entropy",
):
    """
    Aligned with CCD diffllm.generate (block diffusion + low_confidence remasking); records decode_step_map.

    fixed_confidence_topk:
        If integer K>0, each step unmasks ``min(K, row mask count)`` positions by confidence ``x0_p``
        among still-[MASK] tokens (fixed budget). Takes precedence over ``confidence_threshold``.

    confidence_threshold:
        If float (e.g. 0.8), unmask all [MASK] in the current block with ``x0_p`` >= threshold
        (``x0_p`` per alg: default/credit = top-1 prob; topk_margin = margin; entropy = neg-entropy,
        not ideal for 0~1 threshold). If a row still has mask but none meet threshold, fallback to
        argmax confidence among masked positions in the block (1 token). If unset: legacy fixed k/topk
        spread across steps.

    Returns:
        x: (B, prompt_len + gen_length)
        decode_step_map: (B, gen_length) int64 on CPU, -1 = still masked
        finish_step: global_step if early stop, else last inner iteration index (legacy behavior)
        forward_count: number of ``model(...).logits`` forwards in this generation (1 per batch forward)
        entropy_trace: only when ``return_entropy_trace=True``, one entry per forward step.
            ``trace_stat='entropy'``: ``mean_entropy_masked_resp`` (mean Shannon entropy);
            ``trace_stat='maxprob'``: ``mean_max_prob_masked_resp`` (mean top-1 prob);
            ``trace_stat='pmax_residual'``: after ``transfer_index``, ``pmax_decode_residual``
            (max top-1 at decoded positions minus mean top-1 at other masked positions);
            ``trace_stat='decode_entropy_residual'``: ``entropy_decode_residual``
            (mean H on other masked minus mean H on transferred; matches Dream);
            ``trace_stat='both_residuals'``: both residuals in one step (decoded top-1 max minus other mean).
            ``masked_entropy_var`` / ``masked_maxprob_var``: per-row population variance of H or top-1
            on remaining masked response positions, mean over rows with >=2 masks; also
            ``var_*_masked_pre_first_eos_resp`` (only mask before leftmost EOS; equals full band if no EOS yet).
            ``both_masked_vars``: all four variance fields in one step (full band + pre-first-EOS).
            All use raw-logits softmax without extra temperature / top-p (independent of ``alg`` confidence).

        First EOS timing (for post-processing variance past EOS): call
        ``first_eos_response_meta(x, decode_step_map, prompt_len, gen_length, eos_token_id)``;
        ``first_eos_decode_step`` uses the same ``global_step`` as ``entropy_trace[*]['step']``.
    """
    alg = normalize_llada_alg(alg)
    ts = normalize_trace_stat(trace_stat)
    device = prompt.device
    bsz, prompt_len = prompt.shape[0], prompt.shape[1]

    x = torch.full((bsz, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone()

    attention_mask = torch.ones((bsz, prompt_len + gen_length), dtype=torch.long, device=device)

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    inner_steps = steps // num_blocks

    forward_count = 0
    decode_step_map = torch.full((bsz, gen_length), -1, dtype=torch.long, device="cpu")
    global_step = 0
    last_inner_i = 0
    entropy_trace: List[Dict[str, Any]] = [] if return_entropy_trace else []

    _fixed_k: Optional[int] = None
    if fixed_confidence_topk is not None:
        try:
            _fk = int(fixed_confidence_topk)
            _fixed_k = _fk if _fk > 0 else None
        except (TypeError, ValueError):
            _fixed_k = None

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt_len + num_block * block_length : prompt_len + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = (
            get_num_transfer_tokens(block_mask_index, inner_steps)
            if confidence_threshold is None and _fixed_k is None
            else None
        )
        credit_state: Optional[Tuple[torch.Tensor, int]] = None
        block_slice = slice(
            prompt_len + num_block * block_length,
            prompt_len + (num_block + 1) * block_length,
        )

        for i in range(inner_steps):
            last_inner_i = i
            if not (x[:, prompt_len:] == mask_id).any():
                if return_entropy_trace:
                    return x, decode_step_map, global_step, forward_count, entropy_trace
                return x, decode_step_map, global_step, forward_count
            if not (x[:, block_slice] == mask_id).any():
                break
            mask_index = x == mask_id
            logits = model(input_ids=x, attention_mask=attention_mask).logits
            forward_count += 1

            if logits_eos_inf and eos_token_id is not None:
                logits[:, :, eos_token_id] = -torch.inf

            if alg == "credit":
                logits, credit_state = apply_credit_fusion(
                    logits,
                    mask_index,
                    credit_state,
                    credit_alpha=credit_alpha,
                    boost_gamma=credit_boost_gamma,
                    decay_beta=credit_decay_beta,
                )

            if return_entropy_trace and ts in ("entropy", "maxprob"):
                if ts == "maxprob":
                    stat_v, n_mr = _mean_max_prob_llada_resp_masked(
                        logits, mask_index, prompt_len, gen_length
                    )
                    row = {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "maxprob",
                        "mean_max_prob_masked_resp": stat_v,
                        "n_masked_resp": n_mr,
                    }
                else:
                    stat_v, n_mr = _mean_shannon_entropy_llada_resp_masked(
                        logits, mask_index, prompt_len, gen_length
                    )
                    row = {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "entropy",
                        "mean_entropy_masked_resp": stat_v,
                        "n_masked_resp": n_mr,
                    }
                entropy_trace.append(row)

            leos_rel: Optional[int] = None
            if return_entropy_trace and ts in (
                "masked_entropy_var",
                "masked_maxprob_var",
                "both_masked_vars",
            ):
                leos_rel = _leftmost_eos_resp_index(
                    x, prompt_len, gen_length, eos_token_id, batch_row=0
                )

            if return_entropy_trace and ts == "masked_entropy_var":
                v_h, n_r = _var_shannon_entropy_llada_resp_masked(
                    logits, mask_index, prompt_len, gen_length
                )
                v_h_pre, n_r_pre = _var_shannon_entropy_llada_resp_masked(
                    logits,
                    mask_index,
                    prompt_len,
                    gen_length,
                    resp_rel_end_exclusive=leos_rel,
                )
                entropy_trace.append(
                    {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "masked_entropy_var",
                        "var_entropy_masked_resp": v_h,
                        "var_entropy_masked_pre_first_eos_resp": v_h_pre,
                        "n_var_rows": n_r,
                        "n_var_rows_pre_first_eos": n_r_pre,
                    }
                )
            elif return_entropy_trace and ts == "masked_maxprob_var":
                v_p, n_r = _var_max_prob_llada_resp_masked(
                    logits, mask_index, prompt_len, gen_length
                )
                v_p_pre, n_r_pre = _var_max_prob_llada_resp_masked(
                    logits,
                    mask_index,
                    prompt_len,
                    gen_length,
                    resp_rel_end_exclusive=leos_rel,
                )
                entropy_trace.append(
                    {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "masked_maxprob_var",
                        "var_maxprob_masked_resp": v_p,
                        "var_maxprob_masked_pre_first_eos_resp": v_p_pre,
                        "n_var_rows": n_r,
                        "n_var_rows_pre_first_eos": n_r_pre,
                    }
                )
            elif return_entropy_trace and ts == "both_masked_vars":
                v_h, n_rh = _var_shannon_entropy_llada_resp_masked(
                    logits, mask_index, prompt_len, gen_length
                )
                v_p, n_rp = _var_max_prob_llada_resp_masked(
                    logits, mask_index, prompt_len, gen_length
                )
                v_h_pre, n_rh_pre = _var_shannon_entropy_llada_resp_masked(
                    logits,
                    mask_index,
                    prompt_len,
                    gen_length,
                    resp_rel_end_exclusive=leos_rel,
                )
                v_p_pre, n_rp_pre = _var_max_prob_llada_resp_masked(
                    logits,
                    mask_index,
                    prompt_len,
                    gen_length,
                    resp_rel_end_exclusive=leos_rel,
                )
                entropy_trace.append(
                    {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "both_masked_vars",
                        "var_entropy_masked_resp": v_h,
                        "var_maxprob_masked_resp": v_p,
                        "var_entropy_masked_pre_first_eos_resp": v_h_pre,
                        "var_maxprob_masked_pre_first_eos_resp": v_p_pre,
                        "n_var_rows": n_rh,
                        "n_var_rows_pre_first_eos_entropy": n_rh_pre,
                        "n_var_rows_pre_first_eos_maxprob": n_rp_pre,
                    }
                )

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Match build_single_trajectory_llada: EOS -inf on noisy logits only.
            if confidence_eos_eot_inf and eos_token_id is not None:
                logits_with_noise[:, :, eos_token_id] = -torch.inf

            p = F.softmax(logits.float(), dim=-1)

            if alg == "topk_margin":
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                top1_probs = sorted_probs[:, :, 0]
                top2_probs = sorted_probs[:, :, 1]
                x0_p = (top1_probs - top2_probs).to(logits.dtype)
            elif alg == "entropy":
                epsilon = 1e-10
                log_probs = torch.log(p + epsilon)
                x0_p = torch.sum(p * log_probs, dim=-1).to(logits.dtype)
            elif alg in ("default", "credit"):
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1).to(logits.dtype)
            else:
                raise ValueError(f"Unknown alg={alg}; use topk_margin | entropy | default | credit")

            x0_p[:, prompt_len + (num_block + 1) * block_length :] = -torch.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-torch.inf, device=device, dtype=x0_p.dtype))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            if _fixed_k is not None:
                for j in range(confidence.shape[0]):
                    n_mask = int(mask_index[j].sum().item())
                    if n_mask <= 0:
                        continue
                    k = min(_fixed_k, n_mask)
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            elif confidence_threshold is None:
                assert num_transfer_tokens is not None
                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k <= 0:
                        continue
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            else:
                thr = float(confidence_threshold)
                neg_inf = torch.tensor(-float("inf"), device=device, dtype=confidence.dtype)
                for j in range(bsz):
                    sub_mask = x[j, block_slice] == mask_id
                    if not sub_mask.any():
                        continue
                    sub_conf = confidence[j, block_slice]
                    hit = sub_mask & (sub_conf >= thr)
                    if hit.any():
                        transfer_index[j, block_slice] = hit
                    else:
                        masked_conf = torch.where(sub_mask, sub_conf, neg_inf)
                        rel = int(torch.argmax(masked_conf).item())
                        pos = block_slice.start + rel
                        transfer_index[j, pos] = True

            if return_entropy_trace and ts == "both_residuals":
                res_p, n_dp, n_op = _pmax_decode_residual_llada_resp(
                    logits, mask_index, transfer_index, prompt_len, gen_length
                )
                res_e, _, _ = _entropy_decode_residual_llada_resp(
                    logits, mask_index, transfer_index, prompt_len, gen_length
                )
                entropy_trace.append(
                    {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "both_residuals",
                        "pmax_decode_residual": res_p,
                        "entropy_decode_residual": res_e,
                        "n_decode_positions": n_dp,
                        "n_other_masked_positions": n_op,
                    }
                )
            elif return_entropy_trace and ts == "pmax_residual":
                res_v, n_dec, n_oth = _pmax_decode_residual_llada_resp(
                    logits, mask_index, transfer_index, prompt_len, gen_length
                )
                entropy_trace.append(
                    {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "pmax_residual",
                        "pmax_decode_residual": res_v,
                        "n_decode_positions": n_dec,
                        "n_other_masked_positions": n_oth,
                    }
                )
            elif return_entropy_trace and ts == "decode_entropy_residual":
                res_v, n_dec, n_oth = _entropy_decode_residual_llada_resp(
                    logits, mask_index, transfer_index, prompt_len, gen_length
                )
                entropy_trace.append(
                    {
                        "step": global_step,
                        "block": num_block,
                        "inner_i": i,
                        "trace_stat": "decode_entropy_residual",
                        "entropy_decode_residual": res_v,
                        "n_decode_positions": n_dec,
                        "n_other_masked_positions": n_oth,
                    }
                )

            old_x = x.clone()
            x[transfer_index] = x0[transfer_index]

            resp_start = prompt_len
            newly = (old_x == mask_id) & (x != mask_id)
            for bb in range(bsz):
                for pos in range(resp_start, x.shape[1]):
                    if bool(newly[bb, pos].item()):
                        rpos = pos - resp_start
                        if 0 <= rpos < gen_length and decode_step_map[bb, rpos].item() == -1:
                            decode_step_map[bb, rpos] = global_step
            global_step += 1

    if return_entropy_trace:
        return x, decode_step_map, last_inner_i, forward_count, entropy_trace
    return x, decode_step_map, last_inner_i, forward_count


@torch.no_grad()
def llada_block_diffusion_generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    mask_id: int,
    alg: str,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: Optional[int] = None,
    credit_alpha: float = 0.7,
    credit_boost_gamma: float = 0.2,
    credit_decay_beta: float = 0.8,
    confidence_threshold: Optional[float] = None,
    fixed_confidence_topk: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Inference: returns (full token ids (B, prompt_len + gen_length), forward count this generation)."""
    x, _, _, fc, _ = llada_generate_with_tracking(
        model=model,
        prompt=prompt,
        mask_id=mask_id,
        alg=alg,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        eos_token_id=eos_token_id,
        credit_alpha=credit_alpha,
        credit_boost_gamma=credit_boost_gamma,
        credit_decay_beta=credit_decay_beta,
        confidence_threshold=confidence_threshold,
        fixed_confidence_topk=fixed_confidence_topk,
        return_entropy_trace=False,
        trace_stat="entropy",
    )
    return x, fc

#!/usr/bin/env python3
"""
dream_tabom_release training core: ``--teacher groundtruth`` only (mask GT, one forward, CE).

Release entry: ``dream_post/run_tabom_examples.sh`` (torchrun this file or ``train_tabom_llada.py``).
Student: Dream (diffusion) or LLaDA CausalLM (``--student_backend llada``, often via ``train_tabom_llada.py``).

Example::

    python3 dream_post/train_tabom.py \\
        --teacher groundtruth --dataset local --local_data data/prompts.jsonl \\
        --output_dir ./checkpoints/run0
"""

import argparse
import copy
import json
from datetime import timedelta
import logging
import math
import os
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TABOM / DLLM: masked CE on ground-truth responses (dream_tabom_release)."
    )

    # ── Teacher / student models ───────────────────────────────────────────────
    p.add_argument(
        "--teacher",
        default="groundtruth",
        choices=["groundtruth"],
        help="Only groundtruth: mask GT response, single forward, CE on masked positions.",
    )
    p.add_argument(
        "--groundtruth_mask_ratio", type=str, default="0.75",
        help=(
            "[--teacher groundtruth only] Fraction of response positions to mask. "
            "Use a single float (e.g. '0.25') for a fixed ratio, or 'lo:hi' "
            "(e.g. '0.25:0.75') to sample uniformly at random each batch."
        ),
    )
    p.add_argument("--dream_model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument(
        "--student_backend",
        default="dream",
        choices=["dream", "llada"],
        help=(
            "Student architecture: Dream diffusion (default) or LLaDA HF CausalLM+LoRA. "
            "Use llada with --teacher groundtruth; CE uses logits[:,pos,:] without Dream shift."
        ),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--dataset", default="openhermes",
                   choices=["tulu-v2", "openhermes", "local"],
                   help="Dataset to use for prompts")
    p.add_argument("--local_data", default=None,
                   help="Path to local JSONL file for --dataset local")
    p.add_argument(
        "--local_data_method",
        default=None,
        choices=["entropy", "default", "all"],
        help=(
            "When --dataset local: filter JSONL by \\\"method\\\". "
            "entropy: entropy trajectories (Dream TABOM; LLaDA also keeps default+credit). "
            "default: method=default only (LLaDA ling-coder judged JSONL). "
            "all: no filter. "
            "If omitted: entropy for --mask_schedule td, else all."
        ),
    )
    p.add_argument("--off_tokenize", action="store_true",
                   help=(
                       "Skip response re-tokenization for --dataset local + --teacher groundtruth. "
                       "Reads pre-tokenized 'response_token_ids' from the JSONL (as produced by "
                       "build_single_trajectory.py) and feeds them directly to the model, avoiding "
                       "the _get_clean_response_tokens() overhead."
                   ))
    p.add_argument("--max_prompt_len", type=int, default=256,
                   help="Maximum token length of the prompt")

    # ── Generation ────────────────────────────────────────────────────────────
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Response length in Dream's diffusion generation")
    p.add_argument("--gen_steps", type=int, default=128,
                   help="Number of diffusion denoising steps")
    p.add_argument("--temperature", type=float, default=1.0) # need to be high otherwise the model will output very short responses
    p.add_argument("--top_p", type=float, default=0.95)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["q_proj", "v_proj"],
                   help="Linear module names to apply LoRA to")
    # ── Diffusion-style masking & loss reweighting (diffllm-style) ───────────
    p.add_argument(
        "--mask_schedule",
        default="td",
        choices=["fixed", "diffusion", "td"],
        help=(
            "Release schedules: fixed (constant mask ratio); diffusion (random GT mask); "
            "td (TABOM time window + optional entropy_rank_reg)."
        ),
    )

    p.add_argument(
        "--td_supervision_window",
        type=int,
        default=32,
        help=(
            "When --mask_schedule=td: supervised window length M for CE on "
            "[cut, cut+M). Ignored for diffusion/fixed."
        ),
    )

    p.add_argument(
        "--mask_include_padding_resp",
        action="store_true",
        help=(
            "When --teacher groundtruth is used with --mask_schedule fixed|diffusion, "
            "allow masked positions to include response padding slots (i.e., tokens "
            "equal to pad_token_id in the response region). "
            "These masked padding slots are also included in the CE loss mask "
            "(so the model learns to predict pad_token_id at those positions)."
        ),
    )

    p.add_argument(
        "--diffusion_min_t",
        type=float,
        default=0.0,
        help="Lower bound for t when --mask_schedule=diffusion.",
    )
    p.add_argument(
        "--diffusion_max_t",
        type=float,
        default=1.0,
        help="Upper bound for t when --mask_schedule=diffusion.",
    )
    p.add_argument(
        "--entropy_rank_reg",
        action="store_true",
        help=(
            "Entropy-based pairwise ranking regularization (td): "
            "earlier-decoded positions in a local window should have higher negative entropy. "
            "Not used with fixed|diffusion."
        ),
    )
    p.add_argument(
        "--entropy_rank_lambda",
        type=float,
        default=0.1,
        help="Weight for entropy ranking regularization term.",
    )
    p.add_argument(
        "--entropy_rank_window",
        type=int,
        default=32,
        help="Number of earliest decoded positions (by decoding_order) to use in the local ranking window.",
    )
    p.add_argument(
        "--entropy_rank_margin",
        type=float,
        default=0.0,
        help="Hinge margin for pairwise entropy ranking inside the local window.",
    )
    # ── Optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_train_steps", type=int, default=None,
                   help="Hard cap on training steps. If --epochs is set, this is computed automatically.")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of full passes over the dataset. Overrides --max_train_steps when set.")

    # ── Checkpointing / logging ────────────────────────────────────────────────
    p.add_argument("--output_dir", default="./checkpoints/tabom")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument(
        "--log_sample_steps", type=int, default=10,
        help=(
            "Print a decoded training sample (student vs teacher top-K predictions "
            "at masked positions) every N global steps.  0 = disabled."
        ),
    )
    p.add_argument(
        "--log_sample_topk", type=int, default=5,
        help="How many top tokens to show per masked position when --log_sample_steps > 0",
    )
    p.add_argument(
        "--log_sample_max_pos", type=int, default=8,
        help="Maximum number of masked positions to display per sample log",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate gradients over this many micro-batches before an optimizer step. "
                        "Effective batch = batch_size × gradient_accumulation_steps.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strict CUDA reproducibility (default: on): CUBLAS workspace if unset, cudnn.deterministic=True, "
            "cudnn.benchmark=False, TF32 off, torch.use_deterministic_algorithms(warn_only=True). "
            "Pass --no-deterministic for faster runs that may differ run-to-run. "
            "Combine with --seed for repeatable experiments."
        ),
    )
    p.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases")
    p.add_argument("--wandb_project", default="dllm-tabom", help="W&B project name")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name. If not set, wandb auto-generates one.")
    p.add_argument(
        "--wandb_entity", default=None,
        help="W&B entity (your username or team). If not set, uses default from wandb login / WANDB_ENTITY",
    )

    # ── Distributed (single-node multi-GPU via torchrun) ───────────────────────
    p.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "Enable torch.distributed-based multi-GPU training (DDP). "
            "Use with torchrun --nproc_per_node=N. When disabled, runs single-process."
        ),
    )
    p.add_argument(
        "--dist_backend",
        default="nccl",
        help="Distributed backend to use when --distributed (default: nccl).",
    )
    p.add_argument(
        "--dist_url",
        default="env://",
        help="init_method / URL for torch.distributed.init_process_group (default: env://).",
    )
    return p.parse_args()


# ──────────────────────────────── Data ────────────────────────────────────────

# Dream ling_coder TABOM (non_ltr cut): for this JSONL only, --mask_schedule=td samples cut
# from non_ltr_positions decode steps (see load_local_jsonl_items).
LING_CODER_TD_NON_LTR_CUT_BASENAME = "dream_ling_coder.jsonl"


def local_data_uses_ling_coder_td_non_ltr_cut(local_path: str) -> bool:
    return os.path.basename(os.path.normpath(local_path)) == LING_CODER_TD_NON_LTR_CUT_BASENAME


# Each dataset item is a (prompt, response, pre_resp_token_ids,
# non_ltr_positions, decoding_order, data_source) tuple.
# - `response` is None when the dataset does not provide reference answers.
# - `pre_resp_token_ids` is a pre-tokenized response token list (length = max_new_tokens)
#   loaded from the `response_token_ids` field; None when --off_tokenize is not set.
# - `non_ltr_positions` / `decoding_order` are optional lists from trajectories JSONL;
#   they are only populated for local GSM8K-style data built by build_single_trajectory.py.
# - `data_source` is an optional domain tag from JSONL (e.g. prm12, code); None if absent.
DataItem = Tuple[
    str,
    Optional[str],
    Optional[List[int]],
    Optional[List[int]],
    Optional[List[int]],
    Optional[str],
]


class PromptDataset(Dataset):
    def __init__(self, items: List[DataItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DataItem:
        return self.items[idx]


def load_data(args) -> List[DataItem]:
    """Load (prompt, Optional[response], Optional[pre_resp_token_ids]) triples.

    Responses are loaded when available for groundtruth masking; if the dataset
    does not carry responses the second element is None.

    When --off_tokenize is set and the JSONL row contains 'response_token_ids',
    the third element is a List[int] of pre-tokenized response tokens
    (length == max_new_tokens, already padded).  Otherwise the third element is
    None and tokenization happens lazily inside train_step_groundtruth().

    When --local_data_method is set (entropy | default | all), filter JSONL rows by \"method\".
    If omitted: entropy for td, else all. LLaDA + entropy also keeps default+credit.
    """
    off_tokenize: bool = getattr(args, "off_tokenize", False)

    if args.dataset == "tulu-v2":
        from datasets import load_dataset as _ld
        ds = _ld("allenai/tulu-2-sft-mixture", split="train")
        items: List[DataItem] = []
        for ex in ds:
            prompt = response = None
            for msg in ex.get("messages", []):
                if msg["role"] == "user" and prompt is None:
                    prompt = msg["content"]
                elif msg["role"] == "assistant" and response is None:
                    response = msg["content"]
            if prompt is not None:
                items.append((prompt, response, None, None, None, None))
        return items

    if args.dataset == "openhermes":
        from datasets import load_dataset as _ld
        ds = _ld("teknium/OpenHermes-2.5", split="train")
        items = []
        for ex in ds:
            prompt = response = None
            for c in ex.get("conversations", []):
                if c["from"] == "human" and prompt is None:
                    prompt = c["value"]
                elif c["from"] == "gpt" and response is None:
                    response = c["value"]
            if prompt is not None:
                items.append((prompt, response, None, None, None, None))
        return items

    if args.dataset == "local":
        assert args.local_data, "--local_data is required for --dataset local"
        return load_local_jsonl_items(args.local_data, args)

    raise ValueError(f"Unknown dataset: {args.dataset}")


def resolve_local_data_method(args: argparse.Namespace) -> str:
    """entropy | default | all — explicit --local_data_method or implicit from mask_schedule."""
    explicit = getattr(args, "local_data_method", None)
    if explicit is not None:
        return explicit
    if getattr(args, "mask_schedule", "") == "td":
        return "entropy"
    return "all"


def load_local_jsonl_items(local_path: str, args: argparse.Namespace) -> List[DataItem]:
    """Load local JSONL rows into DataItem tuples (same rules as legacy load_data local branch)."""
    off_tokenize: bool = getattr(args, "off_tokenize", False)
    args.ling_coder_td_non_ltr_cut = local_data_uses_ling_coder_td_non_ltr_cut(local_path)
    if args.ling_coder_td_non_ltr_cut:
        logger.info(
            "ling_coder TABOM: %s → td cut sampled from non_ltr_positions when present",
            LING_CODER_TD_NON_LTR_CUT_BASENAME,
        )
    method_filter = resolve_local_data_method(args)
    items: List[DataItem] = []
    with open(local_path) as f:
        for line in f:
            obj = json.loads(line)
            if method_filter == "default":
                if obj.get("method") != "default":
                    continue
            elif method_filter == "entropy":
                m = obj.get("method")
                if m != "entropy":
                    if not (
                        getattr(args, "student_backend", "") == "llada"
                        and m in ("default", "credit")
                    ):
                        continue
            prompt = response = None
            pre_ids: Optional[List[int]] = None
            non_ltr_positions: Optional[List[int]] = None
            decoding_order: Optional[List[int]] = None
            if "prompt" in obj:
                prompt = obj["prompt"]
                response = obj.get("response") or obj.get("output")
            elif "instruction" in obj:
                prompt = obj["instruction"]
                response = obj.get("output") or obj.get("response")
            elif "messages" in obj:
                for msg in obj["messages"]:
                    role = msg.get("role", "")
                    if role == "user" and prompt is None:
                        prompt = msg["content"]
                    elif role == "assistant" and response is None:
                        response = msg["content"]
            elif "question" in obj:
                prompt = obj["question"]
                response = (
                    obj.get("gen_text")
                    or obj.get("answer")
                    or obj.get("output")
                    or obj.get("response")
                )
            if prompt is not None:
                if off_tokenize and "response_token_ids" in obj:
                    pre_ids = obj["response_token_ids"]
                if "non_ltr_positions" in obj:
                    non_ltr_positions = obj["non_ltr_positions"]
                if "decoding_order" in obj:
                    decoding_order = obj["decoding_order"]
            domain: Optional[str] = None
            raw_ds = obj.get("data_source")
            if isinstance(raw_ds, str) and raw_ds.strip():
                domain = raw_ds.strip()
            if prompt is not None:
                items.append((prompt, response, pre_ids, non_ltr_positions, decoding_order, domain))
    logger.info(
        "local (method=%s): loaded %d examples from %s",
        method_filter,
        len(items),
        local_path,
    )
    return items


def tokenize_batch(
    tokenizer,
    prompts: List[str],
    max_prompt_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply chat template + left-pad the batch.  Returns (input_ids, attn_mask).

    Truncation is done by pre-shortening each prompt's *content* before
    applying the chat template, so that the generation-prompt suffix
    (<|im_end|>\\n<|im_start|>assistant\\n) is never accidentally cut off.
    Using apply_chat_template(..., truncation=True) would truncate from the
    right and remove those tokens, leaving the model without a valid assistant
    role marker when the prompt is long.
    """
    # Measure how many tokens the template wrappers consume on their own
    # (system preamble + user role markers + generation-prompt suffix).
    # Use a single space so the tokenizer doesn't treat it as a boundary edge case.
    _overhead_ids: List[int] = tokenizer.apply_chat_template(
        [{"role": "user", "content": " "}],
        add_generation_prompt=True,
    )
    overhead = len(_overhead_ids)
    max_content_tokens = max(1, max_prompt_len - overhead)

    safe_prompts: List[str] = []
    for p in prompts:
        content_ids = tokenizer.encode(p, add_special_tokens=False)
        if len(content_ids) > max_content_tokens:
            content_ids = content_ids[:max_content_tokens]
            p = tokenizer.decode(content_ids, skip_special_tokens=True)
        safe_prompts.append(p)

    messages_list = [[{"role": "user", "content": p}] for p in safe_prompts]
    encoded = tokenizer.apply_chat_template(
        messages_list,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        padding=True,
        truncation=False,   # content already fits; no tail-truncation needed
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


# ──────────── Balanced-sampling / balanced-loss helpers ────────────────────────


def sample_groundtruth_mask_ratio_from_arg(s: str) -> float:
    """Parse a mask-ratio CLI string and return a (possibly random) float.

    Accepted formats: ``"0.75"`` (fixed) or ``"0.25:0.75"`` (uniform sample).
    """
    if isinstance(s, float):
        return s
    s = str(s)
    if ":" in s:
        lo, hi = (float(x) for x in s.split(":", 1))
        return lo + random.random() * (hi - lo)
    return float(s)


def _get_clean_response_tokens(
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize ground-truth responses and pad/truncate to max_new_tokens.

    Returns a (B, max_new_tokens) tensor of response token IDs.  Uses the full
    chat template so that assistant-turn formatting markers (e.g.
    ``<|im_start|>assistant\\n`` prefix and ``<|im_end|>`` suffix) are included
    automatically.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    resp_rows: List[List[int]] = []
    for prompt, response in zip(prompts, responses):
        full_ids: List[int] = tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": response},
            ],
            add_generation_prompt=False,
        )
        prompt_ids: List[int] = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        resp_ids = full_ids[len(prompt_ids):][:max_new_tokens]
        resp_ids += [pad_id] * (max_new_tokens - len(resp_ids))
        resp_rows.append(resp_ids)

    return torch.tensor(resp_rows, dtype=torch.long, device=device)


# ───────────────────────────── Model loading ──────────────────────────────────


def load_llada_student_tabom(
    args, device: torch.device
) -> Tuple[object, object, object, int]:
    """Load GSAI/HF LLaDA CausalLM from ``--dream_model`` and attach LoRA (groundtruth student)."""
    logger.info("Loading LLaDA student (CausalLM) from %s", args.dream_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.dream_model, trust_remote_code=True, padding_side="left"
    )

    load_kw: dict = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if getattr(args, "deterministic", True):
        load_kw["attn_implementation"] = "eager"
    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.dream_model, **load_kw).to(device)
    except TypeError as exc:
        if getattr(args, "deterministic", True) and "attn_implementation" in load_kw:
            load_kw.pop("attn_implementation", None)
            logger.warning(
                "LLaDA load: attn_implementation=eager not accepted (%s); retrying without it.",
                exc,
            )
            base_model = AutoModelForCausalLM.from_pretrained(args.dream_model, **load_kw).to(device)
        else:
            raise

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = getattr(base_model.config, "mask_token_id", None)
    assert mask_id is not None, "Cannot determine LLaDA mask token ID"

    gen_model = base_model

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    base_model.enable_input_require_grads()
    train_model = get_peft_model(base_model, lora_cfg)
    train_model.print_trainable_parameters()

    return train_model, gen_model, tokenizer, int(mask_id)


def load_dream(
    args, device: torch.device
) -> Tuple[object, object, object, int]:
    """Load Dream model + tokenizer; wrap with LoRA for training.

    Returns:
        train_model: PeftModel (Dream + LoRA adapters) — used for student forward.
        gen_model:   Unwrapped base model — used for trajectory collection and,
                     in Dream-privileged mode, for the teacher forward on x_clean.
        tokenizer
        mask_id
    """
    if getattr(args, "student_backend", "dream") == "llada":
        return load_llada_student_tabom(args, device)

    logger.info("Loading Dream base model from  %s", args.dream_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.dream_model, trust_remote_code=True, padding_side="left"
    )

    '''
    # Training ExtendedDreamModel wrapper (avoids lm_eval model registration side effects)
    from Dream.extended_dream_for_train import ExtendedDreamModel
    base_model = ExtendedDreamModel.from_pretrained(
        args.dream_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = getattr(base_model.config, "mask_token_id", None)
    assert mask_id is not None, "Cannot determine Dream mask token ID"

    # gen_model = base copy without LoRA for validation/trajectory (decoupled from train).
    # get_peft_model mutates base_model in place; deepcopy first for gen_model.
    gen_model = copy.deepcopy(base_model)
    gen_model.eval()
    for p in gen_model.parameters():
        p.requires_grad_(False)
    '''

    load_kw: dict = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if getattr(args, "deterministic", True):
        load_kw["attn_implementation"] = "eager"
    try:
        base_model = AutoModel.from_pretrained(args.dream_model, **load_kw).to(device)
    except TypeError as exc:
        if getattr(args, "deterministic", True) and "attn_implementation" in load_kw:
            load_kw.pop("attn_implementation", None)
            logger.warning(
                "Dream load: attn_implementation=eager not accepted (%s); retrying without it.",
                exc,
            )
            base_model = AutoModel.from_pretrained(args.dream_model, **load_kw).to(device)
        else:
            raise

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = getattr(base_model.config, "mask_token_id", None)
    assert mask_id is not None, "Cannot determine Dream mask token ID"

    # Keep a reference to the unwrapped model for trajectory collection.
    # LoRA acts through forward-pass hooks and does NOT modify base weights,
    # so trajectory collection via base_model.diffusion_generate() is
    # equivalent to calling the PEFT model's generation (LoRA → zero initially,
    # small adaptation thereafter).  In Dream-privileged mode gen_model is also
    # used as the frozen teacher.
    gen_model = base_model

    # DreamModel is a diffusion LM and does not implement the standard HF
    # autoregressive interface, so task_type="FEATURE_EXTRACTION" creates a
    # plain PeftModel that only patches the LoRA adapters into the forward
    # pass — generation is done via gen_model.diffusion_generate().
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    base_model.enable_input_require_grads()
    train_model = get_peft_model(base_model, lora_cfg)
    train_model.print_trainable_parameters()

    return train_model, gen_model, tokenizer, mask_id


# ──────────────────────────── Attention helpers ───────────────────────────────

def build_attention_mask_2d(
    prompt_attn: torch.Tensor,   # (B, prompt_len) – 0=pad, 1=real
    full_len: int,
) -> torch.Tensor:
    """Extend prompt attention mask to cover the full (prompt+response) length.

    Response tokens are always real (MASK or already-decoded tokens),
    so they always get attention value 1.
    """
    B = prompt_attn.shape[0]
    resp_len = full_len - prompt_attn.shape[1]
    resp_attn = torch.ones(B, resp_len, dtype=torch.long, device=prompt_attn.device)
    return torch.cat([prompt_attn, resp_attn], dim=1)  # (B, full_len)


def make_4d_mask(attn_2d: torch.Tensor) -> torch.Tensor:
    """Convert 2-D attention mask to 4-D for Dream's bidirectional attention.

    Shape: (B, 1, L, L)  –  True where both query and key positions are real.
    """
    a = attn_2d.bool()
    return (a.unsqueeze(1).unsqueeze(-2) & a.unsqueeze(1).unsqueeze(-1))


def compute_position_ids(attn_2d: torch.Tensor) -> torch.Tensor:
    """Position IDs that respect left-padding (padding slots get id 0)."""
    return (attn_2d.cumsum(dim=1) - 1).clamp(min=0)


# ──────────────────────────── KL loss helpers ─────────────────────────────────

def _align_dream_logits(dream_logits: torch.Tensor) -> torch.Tensor:
    """Apply the causal shift so that aligned[b, pos, :] predicts token at pos.

    Dream's raw convention: raw_logits[b, pos-1, :] predicts token at pos.
    After alignment: aligned[b, pos, :] = raw_logits[b, pos-1, :].
    This is equivalent to diffllm's shift_logits:

        shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
    """
    return torch.cat([dream_logits[:, :1], dream_logits[:, :-1]], dim=1)


# ─────────────────── Training step: groundtruth (normal DLLM) ──────────────────

def train_step_groundtruth(
    train_model,
    optimizer,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    dream_mask_id: int,
    args,
    device: torch.device,
    dream_tokenizer,
    prompts: List[str],
    responses: List[Optional[str]],
    pre_resp_tokens: Optional[torch.Tensor] = None,
    non_ltr_positions_batch: Optional[List[Optional[List[int]]]] = None,
    decoding_order_batch: Optional[List[Optional[List[int]]]] = None,
    do_zero_grad: bool = True,
    do_step: bool = True,
    loss_scale: float = 1.0,
    global_step: Optional[int] = None,
    data_source_batch: Optional[List[Optional[str]]] = None,
) -> float:
    """Normal DLLM training: randomly mask ground-truth response, one forward, CE loss.

    No diffusion_generate and no teacher. Builds [prompt | response], randomly
    masks a fraction of (non-pad) response positions, runs a single forward and
    computes cross-entropy loss on masked positions only.

    When `pre_resp_tokens` is not None (shape (B, max_new_tokens)) it is used
    directly, skipping the _get_clean_response_tokens() tokenization step.
    This is activated by --off_tokenize when the JSONL already carries
    'response_token_ids'.

    `do_zero_grad`, `do_step`, and `loss_scale` are used for gradient accumulation:
    - Set do_zero_grad=False on all but the first micro-batch in an accumulation cycle
      (gradients are kept intact to be summed over micro-batches).
    - Set do_step=False on all but the last micro-batch.
    - Set loss_scale = 1 / gradient_accumulation_steps so each micro-batch contributes
      an equal fraction to the final gradient.
    """
    assert responses is not None, "--teacher groundtruth requires responses"
    B, prompt_len = prompt_input_ids.shape
    if pre_resp_tokens is not None:
        # --off_tokenize path: use pre-built token tensor directly
        clean_resp_tokens = pre_resp_tokens  # (B, max_new_tokens)
    else:
        hints = [r if r is not None else "" for r in responses]
        clean_resp_tokens = _get_clean_response_tokens(
            tokenizer      = dream_tokenizer,
            prompts        = prompts,
            responses      = hints,
            max_new_tokens = args.max_new_tokens,
            device         = device,
        )  # (B, max_new_tokens)
        
    pad_id = dream_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = dream_tokenizer.eos_token_id

    # Full sequence: [prompt | response]
    
    x_full = torch.cat([prompt_input_ids, clean_resp_tokens], dim=1)  # (B, prompt_len + max_new_tokens)
    # Debug print: only on global rank 0 (avoid Tensor.rank which doesn't exist).
    full_len = x_full.shape[1]

    # Which response positions are real (non-pad)?
    resp_region = torch.zeros_like(x_full, dtype=torch.bool)
    resp_region[:, prompt_len:] = True
    non_pad_resp = resp_region & (x_full != pad_id)  # (B, full_len)
    
    mask_include_padding_resp = bool(getattr(args, "mask_include_padding_resp", False))
    # When enabled, we allow input-masking to select padding slots inside the
    # response region; for fixed|diffusion schedules, those padding slots will
    # also be included in the CE loss mask.
    maskable_resp_in = resp_region if mask_include_padding_resp else non_pad_resp


    # ── Masking (release): fixed | diffusion | td ───────────────────────────────
    schedule = getattr(args, "mask_schedule", "fixed")
    # td: per-batch cut for entropy_rank_reg window [cut, cut+W_rank).
    td_cuts_for_entropy: Optional[List[Optional[int]]] = None

    T_resp = args.max_new_tokens

    if schedule == "diffusion":
        # Sample per-example time t in [min_t, max_t].
        min_t = getattr(args, "diffusion_min_t", 0.0)
        max_t = getattr(args, "diffusion_max_t", 1.0)
        t = torch.rand(B, device=device)
        t = min_t + (max_t - min_t) * t  # (B,)

        # Each non-pad response token is masked independently with prob=t[b].
        u = torch.rand_like(x_full, dtype=torch.float, device=device)
        t_mask = (u < t[:, None]) & maskable_resp_in   # True = masked, should incur loss
        x_in = x_full.masked_fill(t_mask, dream_mask_id)
        mask_positions_full = t_mask
    elif schedule == "td":
        # Time-delay: input shows decode_step < cut; CE on [cut, cut+M).
        # dream_ling_coder.jsonl: cut from non_ltr decode steps (data-path trick).
        is_nl_td = bool(getattr(args, "ling_coder_td_non_ltr_cut", False))
        M = max(1, int(getattr(args, "td_supervision_window", 32)))
        td_cuts_for_entropy = [None] * B
        x_in = x_full.clone()
        mask_positions_full = torch.zeros_like(x_full, dtype=torch.bool)
        t = torch.ones(B, device=device, dtype=torch.float32)

        resp_maskable = maskable_resp_in[:, prompt_len : prompt_len + T_resp]  # (B, T_resp)
        resp_slice_all = slice(prompt_len, prompt_len + T_resp)

        for b in range(B):
            decode_order = (
                decoding_order_batch[b]
                if decoding_order_batch is not None and b < len(decoding_order_batch)
                else None
            )
            if decode_order is None or len(decode_order) < T_resp:
                decode_tensor = torch.arange(T_resp, device=device, dtype=torch.long)
            else:
                decode_tensor = torch.tensor(
                    decode_order[:T_resp], device=device, dtype=torch.long
                )

            valid = resp_maskable[b] & (decode_tensor >= 0)
            if not valid.any():
                x_in[b, resp_slice_all] = dream_mask_id
                continue
            s_max = int(decode_tensor[valid].max().item())
            if s_max < M:
                x_in[b, resp_slice_all] = dream_mask_id
                continue

            upper = s_max - M
            if is_nl_td:
                nltr_steps_list: Optional[List[int]] = None
                nltr_pos = None
                if non_ltr_positions_batch is not None and b < len(non_ltr_positions_batch):
                    nltr_pos = non_ltr_positions_batch[b]
                if nltr_pos is not None and decode_order is not None and len(decode_order) >= T_resp:
                    nltr_steps_list = [
                        int(decode_order[p])
                        for p in nltr_pos
                        if 0 <= p < T_resp and int(decode_order[p]) >= 0
                    ]
                if nltr_steps_list:
                    candidates = sorted({s for s in nltr_steps_list if 0 <= s <= upper})
                    if candidates:
                        cut = int(random.choice(candidates))
                    else:
                        cut = int(random.randint(0, upper))
                else:
                    cut = int(random.randint(0, upper))
            else:
                cut = int(random.randint(0, upper))

            td_cuts_for_entropy[b] = cut

            resp_slice = slice(prompt_len, prompt_len + T_resp)
            steps_b = decode_tensor
            # Never-decoded steps (-1) must not count as prefix-unmask.
            unmask_input = (steps_b >= 0) & (steps_b < cut)
            x_in[b, resp_slice] = torch.where(
                unmask_input,
                x_full[b, resp_slice],
                torch.full_like(x_full[b, resp_slice], dream_mask_id),
            )

            in_loss_window = (steps_b >= cut) & (steps_b < cut + M)
            # Align with mask_include_padding_resp: same maskable region as fixed|diffusion CE.
            loss_here = in_loss_window & maskable_resp_in[b, resp_slice]
            if loss_here.any():
                pos = loss_here.nonzero(as_tuple=True)[0] + prompt_len
                mask_positions_full[b, pos] = True

    elif schedule == "fixed":
        mr_global = sample_groundtruth_mask_ratio_from_arg(
            getattr(args, "groundtruth_mask_ratio", "0.75")
        )
        per_b_mr = [mr_global] * B

        x_in = x_full.clone()
        mask_positions_full = torch.zeros_like(x_full, dtype=torch.bool)
        for b in range(B):
            mask_ratio_b = per_b_mr[b]
            positions = maskable_resp_in[b].nonzero(as_tuple=True)[0]
            n = positions.shape[0]
            if n == 0:
                continue
            k = max(1, int(round(n * mask_ratio_b)))
            chosen = positions[torch.randperm(n, device=device)[:k]]
            x_in[b, chosen] = dream_mask_id
            mask_positions_full[b, chosen] = True

        t = torch.tensor(per_b_mr, device=device, dtype=torch.float32)

    else:
        raise ValueError(
            f"Unsupported mask_schedule={schedule!r}; expected fixed|diffusion|td"
        )

    mask_positions_resp = mask_positions_full[:, prompt_len:]  # (B, max_new_tokens)

    if not mask_positions_resp.any():
        return 0.0

    attn_2d = build_attention_mask_2d(prompt_attention_mask, full_len)
    attn_4d = make_4d_mask(attn_2d)
    pos_ids = compute_position_ids(attn_2d)

    if do_zero_grad:
        optimizer.zero_grad()
    if getattr(args, "student_backend", "dream") == "llada":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = train_model(
                input_ids=x_in,
                attention_mask=attn_2d,
                use_cache=False,
            )
        logits = torch.nan_to_num(out.logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        resp_logits = logits[:, prompt_len : prompt_len + T_resp, :]
    else:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = train_model(
                input_ids      = x_in,
                attention_mask = attn_4d,
                position_ids   = pos_ids,
                use_cache      = False,
            )
        logits = torch.nan_to_num(out.logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        aligned = _align_dream_logits(logits)
        resp_logits = aligned[:, prompt_len:, :]  # (B, max_new_tokens, V)

    # ── CE loss at masked positions with optional diffllm-style reweighting ────
    B2, T_resp, V = resp_logits.shape
    assert B2 == B
    # Flatten response region
    logits_flat = resp_logits.reshape(B * T_resp, V)
    labels_flat = clean_resp_tokens.reshape(-1)
    mask_flat = mask_positions_resp.reshape(-1)

    if not mask_flat.any():
        return 0.0

    logits_at_mask = logits_flat[mask_flat]          # (M, V)
    labels_at_mask = labels_flat[mask_flat]          # (M,)
    loss_vec = F.cross_entropy(
        logits_at_mask, labels_at_mask, reduction="none"
    )  # (M,)

    # ── Entropy-based ranking regularization (td, optional) ─────────────────────
    loss_rank = torch.tensor(0.0, device=device)

    if getattr(args, "entropy_rank_reg", False) and schedule == "td":
        # neg_entropy_full: (B, T_resp), keep grad for ranking reg on logits.
        probs_full = torch.softmax(resp_logits, dim=-1)
        log_probs_full = torch.log(probs_full + 1e-10)
        neg_entropy_full = torch.sum(probs_full * log_probs_full, dim=-1)

        rank_terms: List[torch.Tensor] = []
        rank_term_batch_idx: List[int] = []
        W_rank = max(1, int(getattr(args, "entropy_rank_window", 16)))
        margin = float(getattr(args, "entropy_rank_margin", 0.0))

        for b in range(B):
            decode_order = (
                decoding_order_batch[b]
                if decoding_order_batch is not None and b < len(decoding_order_batch)
                else None
            )
            if decode_order is None or len(decode_order) < T_resp:
                if schedule != "td":
                    continue
                steps = torch.arange(T_resp, device=device, dtype=torch.long)
            else:
                steps = torch.tensor(decode_order[:T_resp], device=device, dtype=torch.long)
            mask_b = mask_positions_resp[b]  # (T_resp,)
            valid_mask = (steps >= 0) & mask_b
            if valid_mask.sum().item() < 2:
                continue

            steps_valid = steps[valid_mask].to(torch.float32)           # (T_valid_masked,)
            neg_e_valid = neg_entropy_full[b][valid_mask]               # (T_valid_masked,)

            # Ranking window left edge: td uses sampled cut.
            if (
                schedule == "td"
                and td_cuts_for_entropy is not None
                and td_cuts_for_entropy[b] is not None
            ):
                step_s_b = float(td_cuts_for_entropy[b])
            else:
                step_s_b = float(steps_valid.min().item())

            # Local window: masked positions whose steps fall into [step_s_b, step_s_b + W_rank).
            in_rank_window = (steps_valid >= step_s_b) & (steps_valid < step_s_b + W_rank)
            if not in_rank_window.any():
                continue
            win_idx = torch.where(in_rank_window)[0]
            if win_idx.numel() < 2:
                continue

            steps_win = steps_valid[win_idx]
            neg_e_win = neg_e_valid[win_idx]

            # Sort window positions by step for correct pairwise order.
            order_local = torch.argsort(steps_win)
            steps_ord = steps_win[order_local]
            neg_e_ord = neg_e_win[order_local]
            win_len = int(order_local.numel())

            if win_len >= 2:
                neg_i = neg_e_ord.unsqueeze(1)
                neg_j = neg_e_ord.unsqueeze(0)
                diff_mat = neg_j - neg_i
                triu_mask = torch.triu(torch.ones(win_len, win_len, device=device, dtype=torch.bool), diagonal=1)
                pair_losses = F.relu(diff_mat + margin)[triu_mask]  # (num_pairs,)
                if pair_losses.numel() > 0:
                    rank_terms.append(pair_losses.mean())
                    rank_term_batch_idx.append(b)

        if rank_terms:
            loss_rank = torch.stack(rank_terms).mean()

    loss_ce = loss_vec.mean()
    loss = loss_ce

    if getattr(args, "entropy_rank_reg", False) and schedule == "td" and loss_rank > 0:
        lambda_rank = float(getattr(args, "entropy_rank_lambda", 0.1))
        loss = loss + lambda_rank * loss_rank

    # Optional wandb logging for main vs ranking loss (rank 0 only).
    is_rank0 = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Log per-step losses on global rank 0 only.
        try:
            is_rank0 = torch.distributed.get_rank() == 0
        except Exception:
            is_rank0 = False

    if getattr(args, "wandb", False) and global_step is not None and is_rank0:
        try:
            import wandb  # type: ignore[import]

            # Skip if wandb.init() was not called on this process.
            if getattr(wandb, "run", None) is not None:
                log_dict = {"train/loss_ce_step": float(loss_ce.item())}
                if getattr(args, "entropy_rank_reg", False):
                    log_dict["train/loss_rank_step"] = float(loss_rank.item())
                wandb.log(log_dict, step=global_step)
        except Exception as e:  # noqa: BLE001
            logger.warning("wandb logging for per-step losses failed: %r", e)

    if not torch.isfinite(loss):
        logger.warning("Non-finite loss (%.4g) in groundtruth step — skipping", loss.item())
        optimizer.zero_grad()
        return float("nan")

    (loss * loss_scale).backward()
    return _finish_update(train_model, optimizer, args, loss.unsqueeze(0), 1, do_step=do_step)


def _finish_update(
    train_model,
    optimizer,
    args,
    total_loss: torch.Tensor,
    valid_steps: int,
    do_step: bool = True,
) -> float:
    """Clip gradients, check for non-finite norm, and optionally call optimizer.step().

    When `do_step=False` (gradient accumulation mid-cycle) the gradients are left
    intact so the next micro-batch can add to them; only the scaled loss value is
    returned.
    """
    if not do_step:
        return (total_loss / valid_steps).item()

    trainable  = [p for p in train_model.parameters() if p.requires_grad]
    grad_norm  = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

    if not torch.isfinite(grad_norm):
        logger.warning(
            "Non-finite grad norm (%.4g) — discarding update and zeroing grads",
            grad_norm.item(),
        )
        optimizer.zero_grad()
        return float("nan")

    optimizer.step()
    return (total_loss / valid_steps).item()


def _configure_reproducibility(args, rank: int, is_main_process: bool) -> None:
    """Seed Python / NumPy / PyTorch (CPU + CUDA); by default enable deterministic CUDA ops (see --no-deterministic)."""
    s = int(args.seed)
    random.seed(s)
    torch.manual_seed(s)
    try:
        import numpy as np

        np.random.seed(s)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # SDPA: flash / memory-efficient attention backward can be nondeterministic
            # even with cudnn flags set. Force math SDP before any model load.
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except AttributeError:
                pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if is_main_process:
            sdp_info = ""
            if torch.cuda.is_available():
                try:
                    sdp_info = (
                        f" SDPA: flash={torch.backends.cuda.flash_sdp_enabled()} "
                        f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                        f"math={torch.backends.cuda.math_sdp_enabled()}"
                    )
                except AttributeError:
                    sdp_info = " SDPA: (getter APIs n/a)"
            logger.info(
                "Deterministic mode on: cudnn.deterministic=True, cudnn.benchmark=False, "
                "TF32 disabled, use_deterministic_algorithms(True, warn_only=True where supported)."
                "%s",
                sdp_info,
            )
    elif is_main_process:
        logger.info(
            "Deterministic mode off (--no-deterministic): TF32 / cudnn deterministic / "
            "deterministic algorithms are not forced; expect more run-to-run variance."
        )
    if is_main_process:
        logger.info("RNG seed=%d (CPU/CUDA/NumPy/Python random)", s)


# ──────────────────────────────── Main ────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed setup (single-node DDP via torchrun)
    if args.distributed:
        # torchrun will set these environment variables.
        if not dist.is_initialized():
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                timeout=timedelta(minutes=60),
            )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_distributed = args.distributed and dist.is_initialized()
    is_main_process = (rank == 0)

    _configure_reproducibility(args, rank, is_main_process)

    if is_main_process:
        logger.info("Device: %s | rank=%d | world_size=%d | distributed=%s",
                    device, rank, world_size, is_distributed)
    logger.info("Teacher mode: groundtruth (normal DLLM training)")

    if args.wandb and is_main_process:
        import wandb
        init_kw = {"project": args.wandb_project, "config": vars(args)}
        if args.wandb_entity:
            init_kw["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kw["name"] = args.wandb_run_name
        wandb.init(**init_kw)

    # ── Load models ───────────────────────────────────────────────────────────
    train_model, gen_model, dream_tokenizer, dream_mask_id = load_dream(args, device)

    # Wrap train_model with DDP when requested.
    if is_distributed:
        train_model = DDP(
            train_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    model_for_config = train_model.module if isinstance(train_model, DDP) else train_model

    logger.info(
        "Student vocab=%d mask_id=%d  (groundtruth, no teacher) student_backend=%s",
        model_for_config.config.vocab_size,
        dream_mask_id,
        getattr(args, "student_backend", "dream"),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data from dataset: %s", args.dataset)
    items = load_data(args)
    logger.info("  %d examples loaded", len(items))

    missing = sum(1 for _, r, *_ in items if r is None)
    if missing:
        logger.warning(
            "  %d / %d examples have no ground-truth response; "
            "empty string will be used as fallback for those items.",
            missing, len(items),
        )

    dataset = PromptDataset(items)

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=list,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=list,
        )

    # ── Compute max_train_steps from epochs if not set explicitly ──────────────
    effective_world = world_size if is_distributed else 1
    global_batch_size = args.batch_size * effective_world
    steps_per_epoch = math.ceil(len(items) / global_batch_size)
    if args.epochs is not None and args.max_train_steps is None:
        args.max_train_steps = steps_per_epoch * args.epochs
        if is_main_process:
            logger.info(
                "epochs=%d  steps_per_epoch=%d  max_train_steps=%d  (global_batch_size=%d)",
                args.epochs, steps_per_epoch, args.max_train_steps, global_batch_size,
            )
    elif args.max_train_steps is None:
        args.max_train_steps = 1000
        if is_main_process:
            logger.info("max_train_steps defaulted to 1000 (no epochs set)")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, train_model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    train_model.train()
    global_step  = 0
    running_loss = 0.0
    desc = "TABOM (groundtruth)"
    pbar = tqdm(total=args.max_train_steps, desc=desc, disable=not is_main_process)

    accum_steps  = max(1, args.gradient_accumulation_steps)
    loss_scale   = 1.0 / accum_steps
    micro_idx    = 0   # counts micro-batches within the current accumulation window

    # Epoch cap: --epochs if set, else a large fallback (legacy compat)
    max_epochs = args.epochs if args.epochs is not None else 999_999


    # Single step-0 validation before training (avoid duplicate runs)
    

    for epoch_idx in range(max_epochs):
        if is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch_idx)
        epoch_batches = dataloader

        for raw_batch in epoch_batches:
            if global_step >= args.max_train_steps:
                break

            # raw_batch is List[DataItem]
            prompts_batch      = [item[0] for item in raw_batch]
            responses_batch    = [item[1] for item in raw_batch]
            pre_ids_batch      = [item[2] for item in raw_batch]
            non_ltr_pos_batch  = [item[3] for item in raw_batch]
            decoding_order_bch = [item[4] for item in raw_batch]
            domain_batch       = [item[5] if len(item) > 5 else None for item in raw_batch]

            

            input_ids, attn_mask = tokenize_batch(
                dream_tokenizer, prompts_batch, args.max_prompt_len, device
            )

            # Build pre_resp_tokens tensor when --off_tokenize is active and
            # every item in the batch carries pre-tokenized IDs.
            pre_resp_tokens: Optional[torch.Tensor] = None
            if getattr(args, "off_tokenize", False) and all(x is not None for x in pre_ids_batch):
                max_len = args.max_new_tokens
                pad_id  = dream_tokenizer.pad_token_id or dream_tokenizer.eos_token_id
                rows = []
                for ids in pre_ids_batch:
                    t = ids[:max_len]                        # truncate if longer
                    if len(t) < max_len:                     # right-pad if shorter
                        t = t + [pad_id] * (max_len - len(t))
                    rows.append(t)
                pre_resp_tokens = torch.tensor(rows, dtype=torch.long, device=device)
            

            # Gradient accumulation flags
            is_first_micro = (micro_idx % accum_steps == 0)
            is_last_micro  = (micro_idx % accum_steps == accum_steps - 1)

            loss = train_step_groundtruth(
                train_model           = train_model,
                optimizer             = optimizer,
                prompt_input_ids      = input_ids,
                prompt_attention_mask = attn_mask,
                dream_mask_id         = dream_mask_id,
                args                  = args,
                device                = device,
                dream_tokenizer       = dream_tokenizer,
                prompts               = prompts_batch,
                responses             = responses_batch,
                pre_resp_tokens       = pre_resp_tokens,
                non_ltr_positions_batch = non_ltr_pos_batch,
                decoding_order_batch    = decoding_order_bch,
                do_zero_grad          = is_first_micro,
                do_step               = is_last_micro,
                loss_scale            = loss_scale,
                global_step           = global_step,
                data_source_batch     = domain_batch,
            )

            micro_idx += 1

            # Advance the logical step (optimizer step) only when the
            # accumulation window is complete.
            if not is_last_micro:
                continue

            scheduler.step()

            if not math.isnan(loss):
                running_loss += loss
            global_step  += 1
            if is_main_process:
                pbar.update(1)

            if is_main_process and global_step % args.log_steps == 0:
                avg_loss = running_loss / args.log_steps
                running_loss = 0.0
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                if args.wandb:
                    import wandb
                    wandb.log(
                        {"train/loss": avg_loss, "train/lr": lr},
                        step=global_step,
                    )

            

        # Save per-epoch checkpoint when the epoch had training steps
        if global_step > 0:
            if is_main_process:
                epoch_ckpt = os.path.join(args.output_dir, f"epoch_{epoch_idx + 1}")
                model_to_save = (
                    train_model.module if isinstance(train_model, DDP) else train_model
                )
                model_to_save.save_pretrained(epoch_ckpt)
                dream_tokenizer.save_pretrained(epoch_ckpt)
                logger.info("Saved epoch-%d checkpoint → %s", epoch_idx + 1, epoch_ckpt)

        if global_step >= args.max_train_steps:
            break

    if is_main_process:
        pbar.close()

    if is_main_process:
        final_path = os.path.join(args.output_dir, "final")
        model_to_save = (
            train_model.module if isinstance(train_model, DDP) else train_model
        )
        model_to_save.save_pretrained(final_path)
        dream_tokenizer.save_pretrained(final_path)
        logger.info("Training complete.  Final checkpoint → %s", final_path)

    if args.wandb and is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

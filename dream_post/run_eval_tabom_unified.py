#!/usr/bin/env python
"""
dream_tabom_release unified eval: scan ckpt_root for <tag>/epoch_*/adapter_model.safetensors, or
use --ckpt_paths / --ckpt_paths_file; invokes eval_tabom.sh (eval_tabom_llada.sh when preset ends
with _llada or contains _llada_; override via EVAL_TABOM_SCRIPT).

--preset scans experiment-named dirs under checkpoints/ (advanced); legacy presets removed.
See dream_tabom_release/README.md and run_tabom_release_train_eval.sh.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple

try:
    import wandb  # type: ignore[import]
except Exception:
    wandb = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
DREAM_ROOT = SCRIPT_DIR.parent

# ling_coder LLaDA default-only ckpt dir suffixes (legacy training names)
_LING_CODER_LLADA_PATH_SUFFIX_PRESETS = frozenset(
    {
        "ling_coder_gt_response_random_mask_llada",
        "ling_coder_SD_response_random_mask_bs8_gpu4_llada",
        "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
    }
)

# Default scan root when --preset none (matches run_tabom_release_train_eval Dream TABOM)
_DEFAULT_CKPT_ROOT_NAME = "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu"
_DEFAULT_OUTPUT_ROOT_NAME = "output_prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu"
_DEFAULT_WANDB_PROJECT_NONE = "dllm-tabom-ddp-prm12k-sd-td-random-ent-rank-bs8-g4-eval"

TASK_PRIMARY_METRIC: Dict[str, Tuple[str, str]] = {
    "gsm8k_cot": ("gsm8k_cot", "exact_match,flexible-extract"),
    "humaneval_instruct": ("humaneval_instruct", "pass@1,create_test"),
    "mbpp_instruct": ("mbpp_instruct", "pass_at_1,none"),
    "ifeval": ("ifeval", "prompt_level_loose_acc,none"),
    # lm_eval harness keys: exact_match,none / math_verify,none (matches release results_*.json)
    "minerva_math500": ("minerva_math500", "math_verify,none"),
    # minerva_math = lm_eval MATH 7-subset group (eval_tabom.sh); aggregate in results["minerva_math"]
    "minerva_math": ("minerva_math", "math_verify,none"),
}

PRESET_CHOICES = [
    "none",
    "prm12k_gt_response_random_mask",
    "prm12k_gt_response_random_mask_llada",
    "prm12k_gt_response_random_mask_fixed",
    "ling_coder_gt_response_random_mask",
    "ling_coder_gt_response_random_mask_llada",
    "prm12k_SD_response_random_mask",
    "prm12k_SD_response_random_mask_allmeth_bs8_gpu8",
    "ling_coder_SD_response_random_mask",
    "prm12k_SD_response_random_mask_bs8_gpu4_llada",
    "ling_coder_SD_response_random_mask_bs8_gpu4_llada",
    "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
    "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
    "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
    "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
    "prm12k_SD_response_TD_random_entropy_rank_grid_b32dc_bs8_4gpu_llada",
    "ling_coder_SD_response_TD_non_ltr_cut_entropy_rank_grid_w32_bs8_gpu8_allmeth_lr1e4",
]


def task_eval_subdir(task: str) -> str:
    """Per-task eval subdir under output_root, e.g. humaneval_instruct -> humaneval_instruct_epoch."""
    return f"{task}_epoch"


def default_registry_path(output_root: Path, task: str) -> Path:
    sub = output_root / task_eval_subdir(task)
    if task == "gsm8k_cot":
        return sub / "eval_registry.jsonl"
    short = {
        "humaneval_instruct": "eval_registry_humaneval.jsonl",
        "mbpp_instruct": "eval_registry_mbpp_instruct.jsonl",
        "ifeval": "eval_registry_ifeval.jsonl",
        "minerva_math500": "eval_registry_minerva_math500.jsonl",
        "minerva_math": "eval_registry_minerva_math.jsonl",
    }.get(task, f"eval_registry_{task}.jsonl")
    return sub / short


def apply_preset_namespace(args: argparse.Namespace) -> None:
    """Apply --preset defaults (CLI overrides handled after parse by caller)."""
    name = args.preset
    if name == "none":
        return
    presets: Dict[str, Dict[str, Any]] = {
        "prm12k_gt_response_random_mask": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_gt_response_random_mask",
            "output_root": DREAM_ROOT / "output_prm12k_gt_response_random_mask",
            "wandb_project": "dllm-tabom-ddp-prm12k-gt-diffusion-mask-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "prm12k_gt_response_random_mask",
        },
        "prm12k_gt_response_random_mask_llada": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_gt_response_random_mask_llada",
            "output_root": DREAM_ROOT / "output_prm12k_gt_response_random_mask_llada",
            "wandb_project": "dllm-tabom-ddp-prm12k-gt-diffusion-mask-llada-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "prm12k_gt_response_random_mask_llada",
        },
        "prm12k_gt_response_random_mask_fixed": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_gt_response_random_mask_fixed",
            "output_root": DREAM_ROOT / "output_prm12k_gt_response_random_mask_fixed",
            "wandb_project": "dllm-tabom-ddp-prm12k-gt-fixed-mask-eval",
            "eval_epochs": "1,5",
            "log_subdir": "prm12k_gt_response_random_mask_fixed",
        },
        "ling_coder_gt_response_random_mask": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "ling_coder_gt_response_random_mask",
            "output_root": DREAM_ROOT / "output_ling_coder_gt_response_random_mask",
            "wandb_project": "dllm-tabom-ddp-ling-coder-gt-diffusion-mask-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "ling_coder_gt_response_random_mask",
        },
        "ling_coder_gt_response_random_mask_llada": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "ling_coder_gt_response_random_mask_llada",
            "output_root": DREAM_ROOT / "output_ling_coder_gt_response_random_mask_llada",
            "wandb_project": "dllm-tabom-ddp-ling-coder-gt-diffusion-mask-llada-eval",
            "eval_epochs": "1,5",
            "log_subdir": "ling_coder_gt_response_random_mask_llada",
        },
        "prm12k_SD_response_random_mask": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_SD_response_random_mask",
            "output_root": DREAM_ROOT / "output_prm12k_SD_response_random_mask",
            "wandb_project": "dllm-tabom-ddp-prm12k-sd-diffusion-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "prm12k_SD_response_random_mask",
        },
        "prm12k_SD_response_random_mask_allmeth_bs8_gpu8": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_SD_response_random_mask_allmeth_bs8_gpu8",
            "output_root": DREAM_ROOT / "output_prm12k_SD_response_random_mask_allmeth_bs8_gpu8",
            "wandb_project": "dllm-tabom-ddp-prm12k-sd-diffusion-allmeth-bs8-g8-eval",
            "eval_epochs": "1,5",
            "log_subdir": "prm12k_SD_response_random_mask_allmeth_bs8_gpu8",
        },
        "ling_coder_SD_response_random_mask": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "ling_coder_SD_response_random_mask",
            "output_root": DREAM_ROOT / "output_ling_coder_SD_response_random_mask",
            "wandb_project": "dllm-tabom-ddp-ling-coder-sd-diffusion-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "ling_coder_SD_response_random_mask",
        },
        "prm12k_SD_response_random_mask_bs8_gpu4_llada": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_SD_response_random_mask_bs8_gpu4_llada",
            "output_root": DREAM_ROOT / "output_prm12k_SD_response_random_mask_bs8_gpu4_llada",
            "wandb_project": "dllm-tabom-ddp-prm12k-sd-diffusion-bs8-gpu4-llada-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "prm12k_SD_response_random_mask_bs8_gpu4_llada",
        },
        "ling_coder_SD_response_random_mask_bs8_gpu4_llada": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "ling_coder_SD_response_random_mask_bs8_gpu4_llada",
            "output_root": DREAM_ROOT / "output_ling_coder_SD_response_random_mask_bs8_gpu4_llada",
            "wandb_project": "dllm-tabom-ddp-ling-coder-sd-diffusion-bs8-gpu4-llada-eval",
            "eval_epochs": "1,5",
            "log_subdir": "ling_coder_SD_response_random_mask_bs8_gpu4_llada",
        },
        "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
            "output_root": DREAM_ROOT / "output_prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
            "wandb_project": "dllm-tabom-ddp-prm12k-sd-td-random-ent-rank-bs8-g4-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
        },
        "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
            "output_root": DREAM_ROOT / "output_ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
            "wandb_project": "dllm-tabom-ddp-ling-coder-sd-td-random-ent-rank-bs8-g4-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu",
        },
        "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
            "output_root": DREAM_ROOT / "output_prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
            "wandb_project": "dllm-tabom-ddp-prm12k-sd-td-random-ent-rank-bs8-g4-llada-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
        },
        "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
            "output_root": DREAM_ROOT / "output_ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
            "wandb_project": "dllm-tabom-ddp-ling-coder-sd-td-random-ent-rank-bs8-g4-llada-eval",
            "eval_epochs": "1,5",
            "log_subdir": "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada",
        },
        "prm12k_SD_response_TD_random_entropy_rank_grid_b32dc_bs8_4gpu_llada": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "prm12k_SD_response_TD_random_entropy_rank_grid_b32dc_bs8_4gpu_llada",
            "output_root": DREAM_ROOT / "output_prm12k_SD_response_TD_random_entropy_rank_grid_b32dc_bs8_4gpu_llada",
            "wandb_project": "dllm-tabom-ddp-prm12k-sd-td-random-ent-rank-b32dc-bs8-g4-llada-eval",
            "eval_epochs": "1,5",
            "log_subdir": "prm12k_SD_response_TD_random_entropy_rank_grid_b32dc_bs8_4gpu_llada",
        },
        "ling_coder_SD_response_TD_non_ltr_cut_entropy_rank_grid_w32_bs8_gpu8_allmeth_lr1e4": {
            "ckpt_root": DREAM_ROOT / "checkpoints" / "ling_coder_SD_response_TD_non_ltr_cut_entropy_rank_grid_w32_bs8_gpu8_allmeth_lr1e4",
            "output_root": DREAM_ROOT / "output_ling_coder_SD_response_TD_non_ltr_cut_entropy_rank_grid_w32_bs8_gpu8_allmeth_lr1e4",
            "wandb_project": "dllm-tabom-ddp-ling-coder-sd-td-nlcut-ent-rank-w32-bs8-g8-allmeth-lr1e4-eval",
            "eval_epochs": "1,3,5",
            "log_subdir": "ling_coder_SD_response_TD_non_ltr_cut_entropy_rank_grid_w32_bs8_gpu8_allmeth_lr1e4",
        },
    }
    cfg = presets.get(name)
    if not cfg:
        return
    sfx = os.environ.get("LING_CODER_RUN_DIR_SUFFIX", "").strip()
    if sfx and name in _LING_CODER_LLADA_PATH_SUFFIX_PRESETS:
        wp = cfg.get("wandb_project")
        cfg = {
            **cfg,
            "ckpt_root": Path(str(cfg["ckpt_root"]) + sfx),
            "output_root": Path(str(cfg["output_root"]) + sfx),
            "log_subdir": str(cfg.get("log_subdir", "")) + sfx,
            **({"wandb_project": str(wp) + sfx} if isinstance(wp, str) and wp else {}),
        }
    for k, v in cfg.items():
        if k == "eval_epochs" and args._user_set_eval_epochs:
            continue
        if k == "ckpt_root" and args._user_set_ckpt_root:
            continue
        if k == "output_root" and args._user_set_output_root:
            continue
        if k == "wandb_project" and args._user_set_wandb_project:
            continue
        if k == "out_layout" and args._user_set_out_layout:
            continue
        setattr(args, k, v)


def parse_primary_metric(data: dict, task: str) -> Tuple[str, float]:
    if task not in TASK_PRIMARY_METRIC:
        raise KeyError(f"Unknown task {task!r}. Supported: {sorted(TASK_PRIMARY_METRIC)}")
    sub, metric = TASK_PRIMARY_METRIC[task]
    results = data.get("results")
    if not isinstance(results, dict):
        raise KeyError("Missing or invalid top-level 'results' in eval JSON")
    block = results.get(sub)
    if not isinstance(block, dict) or metric not in block:
        raise KeyError(f"Missing results['{sub}']['{metric}'] in eval JSON (task={task!r})")
    return metric, float(block[metric])


class _TeeTextIO:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._primary.flush()
        self._secondary.write(data)
        self._secondary.flush()
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def fileno(self) -> int:
        return self._primary.fileno()

    def isatty(self) -> bool:
        return self._primary.isatty()


def default_eval_log_path(log_subdir: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = DREAM_ROOT / "logs" / "eval_tabom_unified" / log_subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"eval_unified_{ts}.log"


@dataclass(frozen=True)
class CheckpointInfo:
    ckpt_dir: Path
    tag: str
    epoch: int


def parse_eval_epochs(s: str) -> Set[int]:
    out: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid epoch token {part!r} in --eval_epochs") from e
    if not out:
        raise argparse.ArgumentTypeError("--eval_epochs must contain at least one integer")
    return out


def path_slug(ckpt_dir: Path) -> str:
    s = str(ckpt_dir.resolve())
    h = hashlib.sha1(s.encode()).hexdigest()[:10]
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", ckpt_dir.name)[:80]
    return f"{safe}_{h}"


def checkpoint_from_path(ckpt_dir: Path) -> CheckpointInfo:
    """Build tag/epoch from an adapter root directory.

    Normal layout (same as find_all_checkpoints): ``ckpt_root/<tag>/epoch_N/``.
    Legacy/alternate: ``.../epoch_N/<leaf>`` where the parent folder name is ``epoch_N``.
    """
    ckpt_dir = ckpt_dir.resolve()
    m_dir = re.match(r"^epoch_(\d+)$", ckpt_dir.name)
    if m_dir:
        epoch = int(m_dir.group(1))
        tag = ckpt_dir.parent.name
        return CheckpointInfo(ckpt_dir=ckpt_dir, tag=tag, epoch=epoch)
    parent_name = ckpt_dir.parent.name
    m_parent = re.match(r"^epoch_(\d+)$", parent_name)
    if m_parent:
        epoch = int(m_parent.group(1))
        tag = ckpt_dir.parent.parent.name
        return CheckpointInfo(ckpt_dir=ckpt_dir, tag=tag, epoch=epoch)
    return CheckpointInfo(ckpt_dir=ckpt_dir, tag=ckpt_dir.name, epoch=0)


def load_registry(path: Path) -> Set[str]:
    """Paths that should be skipped when scheduling pending work.

    Only ``status == "done"`` counts as finished. ``failed`` entries are kept
    in the JSONL for audit but do **not** block re-runs so a transient error
    (OOM, timeout, bad PYTHON path) can be retried without editing the file.
    """
    evaluated: Set[str] = set()
    if not path.is_file():
        return evaluated
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ckpt_path = obj.get("ckpt_path")
                status = obj.get("status", "done")
                if isinstance(ckpt_path, str) and status == "done":
                    evaluated.add(ckpt_path)
            except Exception:
                continue
    return evaluated


def load_merged_decode_stats(out_dir: Path) -> Optional[Dict[str, Any]]:
    """Merge multi-GPU diffllm_decode_stats shards into task TPF (decode slots / forwards)."""
    base_name = "diffllm_decode_stats.json"
    shards = sorted(out_dir.glob(f"{base_name}.rank_*"))
    if shards:
        tot_fc = 0
        tot_sl = 0
        for sp in shards:
            try:
                with sp.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                tot_fc += int(d.get("forward_count", 0))
                tot_sl += int(d.get("decode_token_slots", 0))
            except Exception:
                continue
        if tot_fc <= 0:
            return None
        return {
            "forward_count": tot_fc,
            "decode_token_slots": tot_sl,
            "TPF": float(tot_sl) / float(tot_fc),
        }
    p = out_dir / base_name
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        fc = int(d.get("forward_count", 0))
        sl = int(d.get("decode_token_slots", 0))
        if fc <= 0:
            return None
        return {
            "forward_count": fc,
            "decode_token_slots": sl,
            "TPF": float(sl) / float(fc),
        }
    except Exception:
        return None


def append_registry(
    path: Path,
    ckpt: CheckpointInfo,
    status: str,
    score: Optional[float],
    results_path: Optional[Path],
    *,
    task: str,
    metric_key: Optional[str],
    decode_stats: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry: Dict[str, Any] = {
        "ckpt_path": str(ckpt.ckpt_dir),
        "tag": ckpt.tag,
        "epoch": ckpt.epoch,
        "status": status,
        "task": task,
        "metric_key": metric_key,
        "metric_flexible_extract": score,
        "results_path": str(results_path) if results_path is not None else None,
        "timestamp": time.time(),
    }
    if decode_stats:
        entry["TPF"] = decode_stats.get("TPF")
        entry["forward_count"] = decode_stats.get("forward_count")
        entry["decode_token_slots"] = decode_stats.get("decode_token_slots")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def find_all_checkpoints(ckpt_root: Path) -> List[CheckpointInfo]:
    results: List[CheckpointInfo] = []
    if not ckpt_root.is_dir():
        return results
    for tag_dir in sorted(ckpt_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name
        for epoch_dir in sorted(tag_dir.iterdir()):
            if not epoch_dir.is_dir():
                continue
            name = epoch_dir.name
            if not name.startswith("epoch_"):
                continue
            try:
                epoch = int(name.split("_", 1)[1])
            except Exception:
                continue
            adapter_path = epoch_dir / "adapter_model.safetensors"
            if adapter_path.is_file():
                results.append(CheckpointInfo(ckpt_dir=epoch_dir.resolve(), tag=tag, epoch=epoch))
    return results


def parse_tag_hparams(tag: str) -> Dict[str, float]:
    def _to_float_from_tag(s: str) -> float:
        if not s:
            return 0.0
        try:
            n = int(s)
        except Exception:
            return 0.0
        if len(s) == 1:
            return float(n)
        if len(s) == 2:
            return n / 10.0
        if len(s) == 3:
            return n / 100.0
        return float(n)

    parts = tag.split("_")
    if len(parts) < 4:
        return {}
    try:
        W = float(parts[0][1:])
    except Exception:
        W = 0.0
    # Legacy: w{d}_a{d}_lam..._m.._mr..  Release (no ALPHA): w{d}_lam.._m.._mr..
    if len(parts) >= 5 and parts[1].startswith("a"):
        lambda_tag = parts[2].replace("lam", "")
        margin_tag = parts[3].replace("m", "")
        ratio_tag = parts[4].replace("mr", "")
    elif parts[1].startswith("lam"):
        lambda_tag = parts[1].replace("lam", "")
        margin_tag = parts[2].replace("m", "")
        ratio_tag = parts[3].replace("mr", "")
    else:
        return {}
    return {
        "entropy_rank_window": W,
        "entropy_rank_lambda": _to_float_from_tag(lambda_tag),
        "entropy_rank_margin": _to_float_from_tag(margin_tag),
        "groundtruth_mask_ratio": _to_float_from_tag(ratio_tag),
    }


def compute_out_dir(ckpt: CheckpointInfo, args: argparse.Namespace) -> Path:
    base = args.output_root / task_eval_subdir(args.task)
    if args.out_layout == "by_path":
        slug = path_slug(ckpt.ckpt_dir)
        return base / "by_path" / slug
    return base / ckpt.tag / f"epoch_{ckpt.epoch}"


def run_eval_for_checkpoint(
    ckpt: CheckpointInfo,
    args: argparse.Namespace,
) -> Tuple[Optional[float], Optional[Path], str, Optional[Dict[str, Any]]]:
    out_dir = compute_out_dir(ckpt, args)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DIFFLLM_DECODE_STATS_PATH"] = str(out_dir / "diffllm_decode_stats.json")
    if getattr(args, "decode_order_json", None):
        p = Path(args.decode_order_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        env["DECODE_ORDER_JSON"] = str(p)
    elif getattr(args, "write_decode_order", False):
        p = out_dir / "decode_order.json"
        env["DECODE_ORDER_JSON"] = str(p)

    env_eval = os.environ.get("EVAL_TABOM_SCRIPT", "").strip()
    if env_eval:
        eval_script = (SCRIPT_DIR / env_eval).resolve() if not os.path.isabs(env_eval) else Path(env_eval)
    else:
        preset_name = str(getattr(args, "preset", "none") or "none")
        use_llada_eval = preset_name.endswith("_llada") or "_llada_" in preset_name
        eval_script = (
            SCRIPT_DIR / "eval_tabom_llada.sh"
            if use_llada_eval
            else SCRIPT_DIR / "eval_tabom.sh"
        )
    cmd = ["bash", str(eval_script), str(ckpt.ckpt_dir), args.task, str(out_dir)]
    print(f"[eval] ckpt={ckpt.ckpt_dir} -> out={out_dir}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    decode_stats = load_merged_decode_stats(out_dir)
    if decode_stats and decode_stats.get("TPF") is not None:
        print(
            f"[eval] decode TPF={decode_stats['TPF']:.4f} "
            f"(slots={decode_stats['decode_token_slots']}, forwards={decode_stats['forward_count']})"
        )
    if rc != 0:
        print(f"[eval] FAILED for {ckpt.ckpt_dir}: exit code {rc}")
        return None, None, "", decode_stats

    results_files = sorted(out_dir.rglob("results_*.json"))
    if not results_files:
        print(f"[eval] No results_*.json found under {out_dir}")
        return None, None, "", decode_stats

    results_path = max(results_files, key=lambda p: p.stat().st_mtime)
    try:
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        metric_key, score_f = parse_primary_metric(data, args.task)
        print(f"[eval] {args.task} {metric_key}={score_f:.6f} for {ckpt.tag} epoch={ckpt.epoch}")
        return score_f, results_path, metric_key, decode_stats
    except Exception as e:
        print(f"[eval] Failed to parse results from {results_path}: {e}")
        exp = TASK_PRIMARY_METRIC.get(args.task, ("", ""))[1]
        return None, results_path, exp, decode_stats


def ensure_wandb(args: argparse.Namespace) -> Optional[Any]:
    if wandb is None:
        print("[wandb] wandb is not available; will skip wandb logging.")
        return None
    if wandb.run is not None:
        return wandb.run
    init_kw: Dict[str, Any] = {"project": args.wandb_project}
    if args.wandb_entity:
        init_kw["entity"] = args.wandb_entity
    run = wandb.init(**init_kw)
    return run


def log_to_wandb(
    ckpt: CheckpointInfo,
    score: float,
    hparams: Dict[str, float],
    results_path: Optional[Path],
    *,
    task: str,
    metric_key: str,
    decode_stats: Optional[Dict[str, Any]] = None,
) -> None:
    if wandb is None or wandb.run is None:
        return
    log_dict: Dict[str, Any] = {
        "eval/primary_metric": score,
        "eval/epoch": ckpt.epoch,
        "eval/tag": ckpt.tag,
        "eval/task": task,
        "eval/metric_key": metric_key,
    }
    if decode_stats and decode_stats.get("TPF") is not None:
        log_dict["eval/TPF"] = float(decode_stats["TPF"])
        log_dict["eval/forward_count"] = int(decode_stats["forward_count"])
        log_dict["eval/decode_token_slots"] = int(decode_stats["decode_token_slots"])
    if task == "gsm8k_cot":
        log_dict["eval/gsm8k_flexible_extract"] = score
    elif task == "humaneval_instruct":
        log_dict["eval/humaneval_pass_at_1"] = score
    elif task == "mbpp_instruct":
        log_dict["eval/mbpp_pass_at_1"] = score
    elif task == "ifeval":
        log_dict["eval/ifeval_prompt_level_loose"] = score
    elif task == "minerva_math500":
        log_dict["eval/minerva_math500_math_verify"] = score
    elif task == "minerva_math":
        log_dict["eval/minerva_math_math_verify"] = score
    for k, v in hparams.items():
        log_dict[f"hparams/{k}"] = v
    if results_path is not None:
        log_dict["eval/results_path"] = str(results_path)
    wandb.log(log_dict)


def collect_checkpoints(args: argparse.Namespace) -> List[CheckpointInfo]:
    if args.explicit_paths_resolved:
        infos = [checkpoint_from_path(p) for p in args.explicit_paths_resolved]
        return infos
    return find_all_checkpoints(Path(args.ckpt_root))


def filter_by_epochs(ckpts: List[CheckpointInfo], allowed: Set[int]) -> List[CheckpointInfo]:
    out: List[CheckpointInfo] = []
    for c in ckpts:
        if c.epoch == 0:
            if 0 in allowed:
                out.append(c)
        elif c.epoch in allowed:
            out.append(c)
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified eval driver for eval_tabom.sh + registry + wandb.",
    )
    p.add_argument("--preset", type=str, default="none", choices=PRESET_CHOICES)
    p.add_argument(
        "--ckpt_root",
        type=Path,
        default=None,
        help=f"Scan root for <tag>/epoch_*/adapter_model.safetensors (default: checkpoints/{_DEFAULT_CKPT_ROOT_NAME} when --preset none).",
    )
    p.add_argument("--output_root", type=Path, default=None)
    p.add_argument("--task", type=str, default="gsm8k_cot", choices=sorted(TASK_PRIMARY_METRIC))
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--eval_epochs", type=str, default="1,3,5")
    p.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated tag dirs under ckpt_root; empty = all.",
    )
    p.add_argument("--ckpt_paths", type=str, default="", help="Comma-separated checkpoint dirs (explicit mode).")
    p.add_argument("--ckpt_paths_file", type=Path, default=None, help="One checkpoint path per line.")
    p.add_argument(
        "--out_layout",
        type=str,
        default="tag_epoch",
        choices=["tag_epoch", "by_path"],
        help="tag_epoch: output_root/<task>_epoch/<tag>/epoch_N; by_path: .../<task>_epoch/by_path/<slug>.",
    )
    p.add_argument("--sleep_seconds", type=int, default=300)
    p.add_argument("--once", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--no-log-file", action="store_true")
    p.add_argument("--log_subdir", type=str, default="default")
    p.add_argument(
        "--decode_order_json",
        type=Path,
        default=None,
        help="If set, export DECODE_ORDER_JSON to this fixed path for every eval (overrides --write_decode_order).",
    )
    p.add_argument(
        "--write_decode_order",
        action="store_true",
        help="Per-eval set DECODE_ORDER_JSON to <out_dir>/decode_order.json.",
    )
    p.add_argument(
        "--export_run_summary_json",
        type=Path,
        default=None,
        help="After each round, append a JSON summary of processed checkpoints.",
    )
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = p.parse_args(argv)

    args._user_set_ckpt_root = "--ckpt_root" in raw_argv
    args._user_set_output_root = "--output_root" in raw_argv
    args._user_set_wandb_project = "--wandb_project" in raw_argv
    args._user_set_registry = "--registry" in raw_argv
    args._user_set_eval_epochs = "--eval_epochs" in raw_argv
    args._user_set_out_layout = "--out_layout" in raw_argv

    if args.ckpt_root is None:
        args.ckpt_root = DREAM_ROOT / "checkpoints" / _DEFAULT_CKPT_ROOT_NAME
    if args.output_root is None:
        args.output_root = DREAM_ROOT / _DEFAULT_OUTPUT_ROOT_NAME
    if args.wandb_project is None:
        args.wandb_project = _DEFAULT_WANDB_PROJECT_NONE

    apply_preset_namespace(args)

    if args.registry is None:
        args.registry = default_registry_path(Path(args.output_root), args.task)

    explicit: List[Path] = []
    if args.ckpt_paths.strip():
        for part in args.ckpt_paths.split(","):
            part = part.strip()
            if part:
                explicit.append(Path(part).expanduser())
    if args.ckpt_paths_file is not None:
        fp = args.ckpt_paths_file.expanduser()
        if fp.is_file():
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    explicit.append(Path(line).expanduser())

    args.explicit_paths_resolved: Optional[List[Path]] = None
    if explicit:
        args.explicit_paths_resolved = [p.resolve() for p in explicit if p.is_dir()]
        if not args._user_set_out_layout:
            args.out_layout = "by_path"
    else:
        args.explicit_paths_resolved = None

    args.eval_epoch_set = parse_eval_epochs(args.eval_epochs)
    raw_tags = (args.tags or "").strip()
    if raw_tags:
        args.tag_filter: Optional[Set[str]] = {t.strip() for t in raw_tags.split(",") if t.strip()}
    else:
        args.tag_filter = None

    return args


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    allowed = args.eval_epoch_set

    log_fp: Optional[TextIO] = None
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr

    log_subdir = getattr(args, "log_subdir", "default")
    if not args.no_log_file:
        log_path = args.log_file if args.log_file is not None else default_eval_log_path(str(log_subdir))
        log_fp = log_path.open("a", encoding="utf-8", buffering=1)
        log_fp.write(f"\n===== eval unified start {datetime.datetime.now().isoformat()} =====\n")
        log_fp.flush()
        sys.stdout = _TeeTextIO(saved_stdout, log_fp)
        sys.stderr = _TeeTextIO(saved_stderr, log_fp)
        print(f"[unified] Logging to {log_path}")

    try:
        print(f"[unified] ckpt_root={args.ckpt_root}")
        print(f"[unified] output_root={args.output_root}")
        print(f"[unified] registry={args.registry}")
        print(f"[unified] task={args.task}")
        print(f"[unified] eval_epochs={sorted(allowed)} out_layout={args.out_layout}")
        if args.tag_filter is not None:
            print(f"[unified] tags filter={sorted(args.tag_filter)}")
        if args.explicit_paths_resolved:
            print(f"[unified] explicit checkpoints={len(args.explicit_paths_resolved)}")

        run = ensure_wandb(args)
        if run is not None:
            print(f"[wandb] Using run: {run.name} ({run.id})")

        while True:
            evaluated = load_registry(args.registry)
            all_ckpts = collect_checkpoints(args)
            if args.tag_filter is not None and not args.explicit_paths_resolved:
                all_ckpts = [c for c in all_ckpts if c.tag in args.tag_filter]
            all_ckpts = filter_by_epochs(all_ckpts, allowed)

            pending = [c for c in all_ckpts if str(c.ckpt_dir) not in evaluated]

            if not pending:
                print("[unified] No pending checkpoints to evaluate.")
            else:
                print(f"[unified] Found {len(pending)} pending checkpoints.")

            round_entries: List[Dict[str, Any]] = []
            for ckpt in pending:
                hparams = parse_tag_hparams(ckpt.tag)
                score, results_path, metric_key, decode_stats = run_eval_for_checkpoint(ckpt, args)
                status = "done" if score is not None else "failed"
                append_registry(
                    args.registry,
                    ckpt,
                    status=status,
                    score=score,
                    results_path=results_path,
                    task=args.task,
                    metric_key=metric_key or None,
                    decode_stats=decode_stats,
                )
                re_entry: Dict[str, Any] = {
                    "ckpt_path": str(ckpt.ckpt_dir),
                    "tag": ckpt.tag,
                    "epoch": ckpt.epoch,
                    "status": status,
                    "metric_key": metric_key or None,
                    "score": score,
                    "results_path": str(results_path) if results_path else None,
                }
                if decode_stats:
                    re_entry["TPF"] = decode_stats.get("TPF")
                    re_entry["forward_count"] = decode_stats.get("forward_count")
                    re_entry["decode_token_slots"] = decode_stats.get("decode_token_slots")
                round_entries.append(re_entry)
                if score is not None and metric_key:
                    log_to_wandb(
                        ckpt,
                        score,
                        hparams,
                        results_path,
                        task=args.task,
                        metric_key=metric_key,
                        decode_stats=decode_stats,
                    )

            if args.export_run_summary_json and round_entries:
                summary = {
                    "timestamp": time.time(),
                    "iso": datetime.datetime.now().isoformat(),
                    "task": args.task,
                    "registry": str(args.registry),
                    "entries": round_entries,
                }
                args.export_run_summary_json.parent.mkdir(parents=True, exist_ok=True)
                with args.export_run_summary_json.open("a", encoding="utf-8") as sf:
                    sf.write(json.dumps(summary, ensure_ascii=False) + "\n")

            if args.once:
                break

            print(f"[unified] Sleeping for {args.sleep_seconds} seconds before next scan...")
            try:
                time.sleep(args.sleep_seconds)
            except KeyboardInterrupt:
                print("[unified] Interrupted; exiting.")
                break
    finally:
        if log_fp is not None:
            try:
                log_fp.write(f"\n===== eval unified end {datetime.datetime.now().isoformat()} =====\n")
                log_fp.flush()
            except Exception:
                pass
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            log_fp.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Wrapper for run_eval_tabom_unified.py (batch lm_eval over checkpoint trees).
# Env: PRESET, TASK, ONCE, EVAL_EPOCHS, CKPT_ROOT, OUTPUT_ROOT, CKPT_PATHS, ...
# See: python3 dream_post/run_eval_tabom_unified.py --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"
UNIFIED="${SCRIPT_DIR}/run_eval_tabom_unified.py"

if [[ -n "${CKPT_SUBDIR:-}" && -z "${PRESET:-}" ]]; then
  CKPT_ROOT="${CKPT_ROOT:-${DREAM_ROOT}/checkpoints/${CKPT_SUBDIR}}"
fi

EXTRA=()
if [[ -n "${PRESET:-}" ]]; then
  EXTRA+=(--preset "$PRESET")
fi
if [[ -n "${CKPT_ROOT:-}" ]]; then
  EXTRA+=(--ckpt_root "$CKPT_ROOT")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  EXTRA+=(--output_root "$OUTPUT_ROOT")
fi
if [[ -n "${REGISTRY:-}" ]]; then
  EXTRA+=(--registry "$REGISTRY")
fi
if [[ -n "${EVAL_EPOCHS:-}" ]]; then
  EXTRA+=(--eval_epochs "$EVAL_EPOCHS")
fi
if [[ -n "${CKPT_TAGS:-}" ]]; then
  EXTRA+=(--tags "$CKPT_TAGS")
fi
if [[ -n "${CKPT_PATHS:-}" ]]; then
  EXTRA+=(--ckpt_paths "$CKPT_PATHS")
fi
if [[ -n "${CKPT_PATHS_FILE:-}" ]]; then
  EXTRA+=(--ckpt_paths_file "$CKPT_PATHS_FILE")
fi
if [[ -n "${SLEEP_SECONDS:-}" ]]; then
  EXTRA+=(--sleep_seconds "$SLEEP_SECONDS")
fi
if [[ "${ONCE:-0}" == "1" ]]; then
  EXTRA+=(--once)
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA+=(--wandb_project "$WANDB_PROJECT")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  EXTRA+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ "${NO_LOG_FILE:-0}" == "1" ]]; then
  EXTRA+=(--no-log-file)
fi
if [[ -n "${LOG_FILE:-}" ]]; then
  EXTRA+=(--log-file "$LOG_FILE")
fi
if [[ -n "${EXPORT_RUN_SUMMARY_JSON:-}" ]]; then
  EXTRA+=(--export_run_summary_json "$EXPORT_RUN_SUMMARY_JSON")
fi
if [[ "${WRITE_DECODE_ORDER:-0}" == "1" ]]; then
  EXTRA+=(--write_decode_order)
fi
if [[ -n "${DECODE_ORDER_JSON_PATH:-}" ]]; then
  EXTRA+=(--decode_order_json "$DECODE_ORDER_JSON_PATH")
fi
if [[ -n "${OUT_LAYOUT:-}" ]]; then
  EXTRA+=(--out_layout "$OUT_LAYOUT")
fi
if [[ -n "${LOG_SUBDIR:-}" ]]; then
  EXTRA+=(--log_subdir "$LOG_SUBDIR")
fi

TASK="${TASK:-gsm8k_cot}"

exec "$PYTHON" "$UNIFIED" "${EXTRA[@]}" --task "$TASK" "$@"

#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# TABOM release: train / Dream infer / lm_eval per recipe (separate steps).
# Training: torchrun -> train_tabom.py or train_tabom_llada.py (fixed hparams).
#
#   bash dream_post/run_tabom_examples.sh show
#   bash dream_post/run_tabom_examples.sh <recipe> train|eval|infer
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_DREAM="${SCRIPT_DIR}/train_tabom.py"
TRAIN_LLADA="${SCRIPT_DIR}/train_tabom_llada.py"
EVAL_DREAM_SH="${SCRIPT_DIR}/eval_tabom.sh"
EVAL_LLADA_SH="${SCRIPT_DIR}/eval_tabom_llada.sh"
INFER_PY="${SCRIPT_DIR}/infer_tabom.py"

TORCHRUN="${TORCHRUN:-torchrun}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<EOF
TABOM recipes — train / infer / eval (see README.md)

  cd ${DREAM_ROOT}
  bash dream_post/run_tabom_examples.sh show
  bash dream_post/run_tabom_examples.sh <recipe> {train|eval|infer}

recipe: dream_ling_coder | dream_prm12k | llada_prm12k | llada_ling_coder

Optional env: CUDA_VISIBLE_DEVICES, NPROC_PER_NODE, TRAIN_BATCH_SIZE, EPOCHS,
  TRAIN_OUTPUT_DIR, USE_WANDB=1, DRY_RUN=1
EOF
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] '; printf '%q ' "$@"; echo
    return 0
  fi
  (cd "$DREAM_ROOT" && "$@")
}

resolve_recipe() {
  local r="$1"
  RECIPE="$r"
  case "$r" in
    dream_ling_coder|ling_coder_dream|1)
      RECIPE_LABEL="[1] Dream + ling_coder TABOM"
      STUDENT=dream
      TRAIN_ENTRY="$TRAIN_DREAM"
      DATA_REL="dream_post/data/dream_ling_coder.jsonl"
      LOCAL_DATA_METHOD=all
      CKPT_REL="checkpoints/dream_ling_coder"
      EVAL_TASK=mbpp_instruct
      TRAIN_LR=1e-4
      EPOCHS_DEF=5
      NPROC_DEF=8
      CUDA_DEF="0,1,2,3,4,5,6,7"
      ENT_LAMBDA=2
      ENT_MARGIN=0.1
      ENT_WINDOW=32
      TD_WINDOW=32
      MASK_RATIO=0.8
      ;;
    dream_prm12k|prm12k_dream|2)
      RECIPE_LABEL="[2] Dream + prm12k TABOM"
      STUDENT=dream
      TRAIN_ENTRY="$TRAIN_DREAM"
      DATA_REL="dream_post/data/dream_prm12k.jsonl"
      LOCAL_DATA_METHOD=entropy
      CKPT_REL="checkpoints/dream_prm12k"
      EVAL_TASK=gsm8k_cot
      TRAIN_LR=2e-5
      EPOCHS_DEF=5
      NPROC_DEF=4
      CUDA_DEF="0,1,2,3"
      ENT_LAMBDA=1
      ENT_MARGIN=0.1
      ENT_WINDOW=32
      TD_WINDOW=32
      MASK_RATIO=0.8
      ;;
    llada_prm12k|prm12k_llada|3)
      RECIPE_LABEL="[3] LLaDA + prm12k TABOM"
      STUDENT=llada
      TRAIN_ENTRY="$TRAIN_LLADA"
      DATA_REL="dream_post/data/llada_prm12k.jsonl"
      LOCAL_DATA_METHOD=entropy
      CKPT_REL="checkpoints/llada_prm12k"
      EVAL_TASK=gsm8k_cot
      TRAIN_LR=2e-5
      EPOCHS_DEF=5
      NPROC_DEF=4
      CUDA_DEF="0,1,2,3"
      ENT_LAMBDA=1
      ENT_MARGIN=0.2
      ENT_WINDOW=32
      TD_WINDOW=32
      MASK_RATIO=0.8
      ;;
    llada_ling_coder|ling_coder_llada|4)
      RECIPE_LABEL="[4] LLaDA + ling_coder TABOM (method=default)"
      STUDENT=llada
      TRAIN_ENTRY="$TRAIN_LLADA"
      DATA_REL="dream_post/data/llada_ling_coder.jsonl"
      LOCAL_DATA_METHOD=default
      CKPT_REL="checkpoints/llada_ling_coder"
      EVAL_TASK=humaneval_instruct
      TRAIN_LR=2e-5
      EPOCHS_DEF=5
      NPROC_DEF=4
      CUDA_DEF="0,1,2,3"
      ENT_LAMBDA=1
      ENT_MARGIN=0.3
      ENT_WINDOW=32
      TD_WINDOW=32
      MASK_RATIO=0.8
      ;;
    *)
      echo "ERROR: unknown recipe=${r}" >&2
      usage >&2
      exit 1
      ;;
  esac
  CKPT_ABS="${DREAM_ROOT}/${CKPT_REL}"
  EPOCHS="${EPOCHS:-${EPOCHS_DEF}}"
  NPROC="${NPROC_PER_NODE:-${NPROC_DEF}}"
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEF}}"
  PER_GPU_BATCH="${TRAIN_BATCH_SIZE:-8}"
  TRAIN_OUT="${TRAIN_OUTPUT_DIR:-checkpoints/${RECIPE}/train_${RUN_TS}}"
}

build_train_argv() {
  TRAIN_ARGV=(
    --dataset local
    --local_data "${DATA_REL}"
    --local_data_method "${LOCAL_DATA_METHOD}"
    --teacher groundtruth
    --off_tokenize
    --mask_schedule td
    --mask_include_padding_resp
    --td_supervision_window "${TD_WINDOW}"
    --entropy_rank_reg
    --entropy_rank_window "${ENT_WINDOW}"
    --entropy_rank_lambda "${ENT_LAMBDA}"
    --entropy_rank_margin "${ENT_MARGIN}"
    --groundtruth_mask_ratio "${MASK_RATIO}"
    --lora_rank 16
    --lora_alpha 16
    --batch_size "${PER_GPU_BATCH}"
    --epochs "${EPOCHS}"
    --learning_rate "${TRAIN_LR}"
    --distributed
    --output_dir "${TRAIN_OUT}"
  )
  if [[ "${USE_WANDB:-0}" == "1" ]]; then
    TRAIN_ARGV+=(--wandb --wandb_project "tabom-${RECIPE}" --wandb_run_name "${RECIPE}-${RUN_TS}")
  fi
}

train_cmd_line() {
  build_train_argv
  echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${TORCHRUN} --standalone --nproc_per_node=${NPROC} \\"
  echo "  ${TRAIN_ENTRY#${DREAM_ROOT}/} \\"
  local i
  for ((i = 0; i < ${#TRAIN_ARGV[@]}; i++)); do
    echo "  ${TRAIN_ARGV[$i]} \\"
  done | sed '$ s/ \\$//'
}

print_block() {
  local eval_sh infer_block student_label
  student_label=$([[ "$STUDENT" == dream ]] && echo Dream || echo LLaDA)
  if [[ "$STUDENT" == dream ]]; then
    eval_sh="bash dream_post/eval_tabom.sh ${CKPT_REL} ${EVAL_TASK} output_eval_examples/${RECIPE}/${EVAL_TASK}"
    infer_block="python dream_post/infer_tabom.py --checkpoint ${CKPT_REL} --prompt \"...\""
  else
    eval_sh="bash dream_post/eval_tabom_llada.sh ${CKPT_REL} ${EVAL_TASK} output_eval_examples/${RECIPE}/${EVAL_TASK}"
    infer_block="(LLaDA: use eval_tabom_llada.sh; no standalone infer CLI in this package)"
  fi

  cat <<EOF

================================================================================
${RECIPE_LABEL}
================================================================================
  recipe          : ${RECIPE}
  student         : ${student_label}
  train data      : ${DATA_REL}
  bundled ckpt    : ${CKPT_REL}
  lm_eval task    : ${EVAL_TASK}

-- Train (DDP -> $(basename "$TRAIN_ENTRY")) --
$(train_cmd_line)

  output: ${TRAIN_OUT}/  (bundled release ckpt: checkpoints/${RECIPE}/)

-- Eval (bundled checkpoint) --
  ${eval_sh}

-- Dream inference --
  ${infer_block}
EOF
}

do_train() {
  build_train_argv
  echo ">>> Train ${RECIPE}"
  echo "    entry: $(basename "$TRAIN_ENTRY")"
  echo "    output: ${TRAIN_OUT}"
  if [[ "$DRY_RUN" == "1" ]]; then
    train_cmd_line
    return 0
  fi
  mkdir -p "${DREAM_ROOT}/logs"
  local log_file="${DREAM_ROOT}/logs/train_${RECIPE}_${RUN_TS}.log"
  run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    "${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" \
    "${TRAIN_ENTRY}" "${TRAIN_ARGV[@]}" \
    >"${log_file}" 2>&1
  echo "log: ${log_file}"
}

do_eval() {
  if [[ ! -f "${CKPT_ABS}/adapter_config.json" ]]; then
    echo "WARN: bundled checkpoint missing or not flattened: ${CKPT_ABS}" >&2
    echo "      run: bash dream_post/rename_release_checkpoints.sh" >&2
  fi
  local out="output_eval_examples/${RECIPE}/${EVAL_TASK}"
  echo ">>> Eval ${RECIPE} -> ${EVAL_TASK}"
  if [[ "$STUDENT" == dream ]]; then
    run_cmd bash "$EVAL_DREAM_SH" "$CKPT_REL" "$EVAL_TASK" "$out"
  else
    run_cmd bash "$EVAL_LLADA_SH" "$CKPT_REL" "$EVAL_TASK" "$out"
  fi
}

do_infer() {
  if [[ "$STUDENT" != dream ]]; then
    echo "infer only supports dream_prm12k / dream_ling_coder" >&2
    exit 1
  fi
  [[ -f "${CKPT_ABS}/adapter_config.json" ]] || {
    echo "WARN: no checkpoint at ${CKPT_ABS} (try rename_release_checkpoints.sh)" >&2
    exit 1
  }
  local prompt="${PROMPT:-Explain what a hash table is.}"
  run_cmd python3 "$INFER_PY" --checkpoint "$CKPT_REL" --prompt "$prompt"
}

show_all() {
  for r in dream_ling_coder dream_prm12k llada_prm12k llada_ling_coder; do
    resolve_recipe "$r"
    print_block
  done
}

main() {
  local recipe="${1:-}"
  local action="${2:-show}"

  case "${recipe}" in
    help|-h|--help|"")
      usage
      exit 0
      ;;
    show|all)
      show_all
      exit 0
      ;;
  esac

  resolve_recipe "$recipe"

  case "${action}" in
    show|print|commands) print_block ;;
    train) do_train ;;
    eval|evaluate|lm_eval) do_eval ;;
    infer|inference) do_infer ;;
    *)
      echo "ERROR: unknown action=${action}" >&2
      exit 1
      ;;
  esac
}

main "$@"

#!/usr/bin/env bash
# Evaluate a TABOM-finetuned Dream checkpoint with lm_eval.
#
# If accelerate/lm_eval fails with:
#   ValueError: numpy.dtype size changed ... pandas._libs.interval
# numpy 2.x may be incompatible with an older pandas wheel; run:
#   pip install "pandas>=2.2.0"
#
# Usage:
#   bash eval_tabom.sh <checkpoint_path> [task] [output_dir]
#
# Examples:
#   bash eval_tabom.sh ./checkpoints/dream_gt_tabom/final
#   bash eval_tabom.sh ./checkpoints/dream_gt_tabom/final gsm8k_cot output_tabom/gsm8k
#   bash eval_tabom.sh ./checkpoints/tabom/final mmlu_generative output_tabom/mmlu
#
# The checkpoint must be a LoRA adapter directory saved by train_tabom.py.
# The base Dream model is always Dream-org/Dream-v0-Instruct-7B.
#
# Optional Dream decode budget (pick one; see diffllm ExtendedDreamModel._sample):
#   export DREAM_PER_STEP_TRANSFER_CAP=2          # unmask K tokens per step
#   export DREAM_CONFIDENCE_THRESHOLD=0.8         # top-1 prob threshold (aligned with LLaDA)

set -euo pipefail

CHECKPOINT="${1:?Usage: bash eval_tabom.sh <checkpoint_path> [task] [output_dir]}"
TASK="${2:-gsm8k_cot}"
OUTPUT_DIR="${3:-output_tabom/$(basename "$CHECKPOINT")/${TASK}}"

BASE_MODEL="Dream-org/Dream-v0-Instruct-7B"

# Per-task defaults (mirror Dream/eval_instruct/eval.sh)
case "$TASK" in
  mmlu_generative)
    MAX_NEW_TOKENS=128; DIFF_STEPS=128; NUM_FEWSHOT=4 ;;
  mmlu_pro)
    MAX_NEW_TOKENS=128; DIFF_STEPS=128; NUM_FEWSHOT=4 ;;
  minerva_math500)
    MAX_NEW_TOKENS=512; DIFF_STEPS=512; NUM_FEWSHOT=0 ;;
  gsm8k_cot)
    MAX_NEW_TOKENS=256; DIFF_STEPS=256; NUM_FEWSHOT=0 ;;
  minerva_math)
    MAX_NEW_TOKENS=512; DIFF_STEPS=512; NUM_FEWSHOT=0 ;;
  gpqa_main_n_shot)
    MAX_NEW_TOKENS=256; DIFF_STEPS=256; NUM_FEWSHOT=5 ;;
  humaneval_instruct)
    MAX_NEW_TOKENS=768; DIFF_STEPS=768; NUM_FEWSHOT=0 ;;
  mbpp_instruct)
    MAX_NEW_TOKENS=512; DIFF_STEPS=512; NUM_FEWSHOT=0 ;;
  ifeval)
    MAX_NEW_TOKENS=512; DIFF_STEPS=512; NUM_FEWSHOT=0 ;;
  *)
    MAX_NEW_TOKENS=256; DIFF_STEPS=256; NUM_FEWSHOT=0 ;;
esac

echo "Checkpoint : $CHECKPOINT"
echo "Base model : $BASE_MODEL"
echo "Task       : $TASK"
echo "Output     : $OUTPUT_DIR"
echo "Steps      : $DIFF_STEPS  |  Max new tokens: $MAX_NEW_TOKENS  |  Few-shot: $NUM_FEWSHOT"
echo

# Use the project's lm_eval (Dream eval_instruct) so that --model diffllm loads the
# version that supports lora_path. Otherwise the pip-installed lm_eval may be used
# and its diffllm might not load LoRA, giving identical results for all checkpoints.
EVAL_LM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${EVAL_LM_ROOT}/dream_post/Dream/eval_instruct:${EVAL_LM_ROOT}"

# Optional decode_order JSON (if diffllm supports decode_order_json_path):
#   export DECODE_ORDER_JSON=/path/to/decode_order.json
DECODE_ORDER_ARG=""
if [[ -n "${DECODE_ORDER_JSON:-}" ]]; then
  DECODE_ORDER_ARG=",decode_order_json_path=${DECODE_ORDER_JSON}"
fi

if [[ -n "${DREAM_CONFIDENCE_THRESHOLD:-}" && -n "${DREAM_PER_STEP_TRANSFER_CAP:-}" ]]; then
  echo "ERROR: set only one of DREAM_CONFIDENCE_THRESHOLD and DREAM_PER_STEP_TRANSFER_CAP" >&2
  exit 1
fi
PER_STEP_CAP_ARG=""
if [[ -n "${DREAM_CONFIDENCE_THRESHOLD:-}" ]]; then
  PER_STEP_CAP_ARG=",dream_confidence_threshold=${DREAM_CONFIDENCE_THRESHOLD}"
elif [[ -n "${DREAM_PER_STEP_TRANSFER_CAP:-}" ]]; then
  PER_STEP_CAP_ARG=",dream_fixed_tokens_per_step=${DREAM_PER_STEP_TRANSFER_CAP}"
fi

# HumanEval / MBPP require code execution
if [[ "$TASK" == "humaneval_instruct" || "$TASK" == "mbpp_instruct" ]]; then
    export HF_ALLOW_CODE_EVAL=1
fi

# Decode stats JSON (run_eval_tabom_unified overrides out_dir when used)
export DIFFLLM_DECODE_STATS_PATH="${DIFFLLM_DECODE_STATS_PATH:-${OUTPUT_DIR}/diffllm_decode_stats.json}"

PYTHONPATH="$PYTHONPATH" accelerate launch --main_process_port 12335 -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${BASE_MODEL},lora_path=${CHECKPOINT},trust_remote_code=True,max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFF_STEPS},dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy${DECODE_ORDER_ARG}${PER_STEP_CAP_ARG}" \
    --tasks "$TASK" \
    --device cuda \
    --batch_size 1 \
    --num_fewshot "$NUM_FEWSHOT" \
    --output_path "$OUTPUT_DIR" \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

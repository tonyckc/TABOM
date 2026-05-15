#!/usr/bin/env bash
# Evaluate LoRA from train_tabom_llada / train_tabom --student_backend llada (lm_eval + diffllm backend=llada).
#
# Usage:
#   bash eval_tabom_llada.sh <checkpoint_path> [task] [output_dir]
#
# Mirrors dream_post/eval_tabom.sh; hparams aligned with eval_baseline_llada.sh.
#
# Optional decode_order JSON (if diffllm supports decode_order_json_path):
#   export DECODE_ORDER_JSON=/path/to/decode_order.json
#
# Optional confidence threshold per step (top-1 prob for default/credit; try 0~1):
#   export LLADA_CONFIDENCE_THRESHOLD=0.8
#
# Optional fixed top-K unmask per step (mutually exclusive with threshold; K wins if both set):
#   export LLADA_FIXED_CONFIDENCE_TOPK=2

set -euo pipefail

CHECKPOINT="${1:?Usage: bash eval_tabom_llada.sh <checkpoint_path> [task] [output_dir]}"
TASK="${2:-gsm8k_cot}"
OUTPUT_DIR="${3:-output_tabom_llada/$(basename "$CHECKPOINT")/${TASK}}"

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12336}"

# Per-task defaults (aligned with eval_baseline_llada.sh)
case "$TASK" in
  gsm8k_cot)
    # export BLOCK_LENGTH=32 to match trajectory blocks (default 8 for legacy reproducibility)
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
    DIFF_STEPS="${DIFF_STEPS:-256}"
    BLOCK_LENGTH="${BLOCK_LENGTH:-8}"
    LOGITS_EOS_INF=False
    CONFIDENCE_EOS_EOT_INF=False
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=llada_original
    NUM_FEWSHOT=0
    ;;
  humaneval_instruct)
    MAX_NEW_TOKENS=512
    DIFF_STEPS=512
    BLOCK_LENGTH=512
    LOGITS_EOS_INF=True
    CONFIDENCE_EOS_EOT_INF=False
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=llada_original
    NUM_FEWSHOT=0
    ;;
  minerva_math|minerva_math500)
    MAX_NEW_TOKENS=512
    DIFF_STEPS=512
    BLOCK_LENGTH=64
    LOGITS_EOS_INF=False
    CONFIDENCE_EOS_EOT_INF=False
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=llada_original
    NUM_FEWSHOT=0
    ;;
  mbpp_instruct)
    MAX_NEW_TOKENS=256
    DIFF_STEPS=256
    BLOCK_LENGTH=256
    LOGITS_EOS_INF=False
    CONFIDENCE_EOS_EOT_INF=True
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=llada_original
    NUM_FEWSHOT=0
    ;;
  mmlu_generative|mmlu_pro)
    MAX_NEW_TOKENS=128
    DIFF_STEPS=128
    BLOCK_LENGTH=128
    LOGITS_EOS_INF=False
    CONFIDENCE_EOS_EOT_INF=False
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=entropy
    NUM_FEWSHOT=4
    ;;
  gpqa_main_n_shot)
    MAX_NEW_TOKENS=256
    DIFF_STEPS=256
    BLOCK_LENGTH=256
    LOGITS_EOS_INF=False
    CONFIDENCE_EOS_EOT_INF=False
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=entropy
    NUM_FEWSHOT=5
    ;;
  ifeval)
    MAX_NEW_TOKENS=512
    DIFF_STEPS=512
    BLOCK_LENGTH=512
    LOGITS_EOS_INF=False
    CONFIDENCE_EOS_EOT_INF=False
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=entropy
    NUM_FEWSHOT=0
    ;;
  *)
    MAX_NEW_TOKENS=256
    DIFF_STEPS=256
    BLOCK_LENGTH=256
    LOGITS_EOS_INF=False
    CONFIDENCE_EOS_EOT_INF=False
    TEMPERATURE=0.
    TOP_P=0.9
    ALG=entropy
    NUM_FEWSHOT=0
    ;;
esac

echo "Checkpoint : $CHECKPOINT"
echo "Base model : $BASE_MODEL"
echo "Task       : $TASK"
echo "Output     : $OUTPUT_DIR"
echo "backend=llada | steps=$DIFF_STEPS | max_new_tokens=$MAX_NEW_TOKENS | block=$BLOCK_LENGTH | fewshot=$NUM_FEWSHOT"
echo

EVAL_LM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${EVAL_LM_ROOT}/dream_post/Dream/eval_instruct:${EVAL_LM_ROOT}"

DECODE_ORDER_ARG=""
if [[ -n "${DECODE_ORDER_JSON:-}" ]]; then
  DECODE_ORDER_ARG=",decode_order_json_path=${DECODE_ORDER_JSON}"
fi

CONF_THRESH_ARG=""
if [[ -n "${LLADA_CONFIDENCE_THRESHOLD:-}" && -z "${LLADA_FIXED_CONFIDENCE_TOPK:-}" ]]; then
  CONF_THRESH_ARG=",confidence_threshold=${LLADA_CONFIDENCE_THRESHOLD}"
fi
FIXED_TOPK_ARG=""
if [[ -n "${LLADA_FIXED_CONFIDENCE_TOPK:-}" ]]; then
  FIXED_TOPK_ARG=",llada_fixed_confidence_topk=${LLADA_FIXED_CONFIDENCE_TOPK}"
fi

if [[ "$TASK" == "humaneval_instruct" || "$TASK" == "mbpp_instruct" ]]; then
    export HF_ALLOW_CODE_EVAL=1
fi

export DIFFLLM_DECODE_STATS_PATH="${DIFFLLM_DECODE_STATS_PATH:-${OUTPUT_DIR}/diffllm_decode_stats.json}"

MODEL_ARGS="pretrained=${BASE_MODEL},lora_path=${CHECKPOINT},trust_remote_code=True,backend=llada,max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFF_STEPS},block_length=${BLOCK_LENGTH},dtype=bfloat16,logits_eos_inf=${LOGITS_EOS_INF},confidence_eos_eot_inf=${CONFIDENCE_EOS_EOT_INF},temperature=${TEMPERATURE},top_p=${TOP_P},alg=${ALG}${DECODE_ORDER_ARG}${CONF_THRESH_ARG}${FIXED_TOPK_ARG}"

PYTHONPATH="$PYTHONPATH" accelerate launch --main_process_port "${MAIN_PROCESS_PORT}" -m lm_eval \
    --model diffllm \
    --model_args "${MODEL_ARGS}" \
    --tasks "$TASK" \
    --device cuda \
    --batch_size 1 \
    --num_fewshot "$NUM_FEWSHOT" \
    --output_path "$OUTPUT_DIR" \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

python3 "${SCRIPT_DIR}/merge_diffllm_decode_stats.py" "${OUTPUT_DIR}" || true

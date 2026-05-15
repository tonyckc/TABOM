#!/usr/bin/env bash
# Train + eval in one go (forwards to run_tabom_examples.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES="${SCRIPT_DIR}/run_tabom_examples.sh"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

usage() {
  cat <<'EOF'
Usage: bash dream_post/run_tabom_release_train_eval.sh <recipe>

recipe: dream_prm12k | dream_ling_coder | llada_prm12k | llada_ling_coder | all

Separate steps: bash dream_post/run_tabom_examples.sh <recipe> train|eval|infer
Env: SKIP_TRAIN=1 | SKIP_EVAL=1
EOF
}

run_recipe() {
  local recipe="$1"
  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    bash "$EXAMPLES" "$recipe" train
  fi
  if [[ "${SKIP_EVAL}" != "1" ]]; then
    bash "$EXAMPLES" "$recipe" eval
  fi
}

dispatch() {
  case "${1:-}" in
    dream_prm12k|prm12k_dream|1) run_recipe dream_prm12k ;;
    dream_ling_coder|ling_coder_dream|2) run_recipe dream_ling_coder ;;
    llada_prm12k|prm12k_llada|3) run_recipe llada_prm12k ;;
    llada_ling_coder|ling_coder_llada|4) run_recipe llada_ling_coder ;;
    all)
      for r in dream_prm12k dream_ling_coder llada_prm12k llada_ling_coder; do
        dispatch "$r"
      done
      ;;
    help|-h|--help|"") usage ;;
    *) echo "ERROR: unknown recipe=$1" >&2; usage >&2; exit 1 ;;
  esac
}

dispatch "$@"

#!/usr/bin/env bash
# Train+eval shortcut (see README.md). For separate steps: bash run_tabom_examples.sh show
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="${ROOT}/dream_post/run_tabom_release_train_eval.sh"

if [[ ! -f "${INNER}" ]]; then
  echo "ERROR: missing ${INNER}" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  echo "TABOM release — see README.md"
  echo ""
  echo "Train+eval: bash run_tabom_recipes.sh <recipe>"
  echo "Separate:   bash run_tabom_examples.sh show"
  exit 0
fi

exec bash "${INNER}" "$@"

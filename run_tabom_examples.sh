#!/usr/bin/env bash
# Forwards to dream_post/run_tabom_examples.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT}/dream_post/run_tabom_examples.sh" "$@"

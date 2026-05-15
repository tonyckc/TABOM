#!/usr/bin/env bash
# Flatten bundled LoRA into checkpoints/<recipe>/ (adapter files at recipe root, no epoch_* subdir).
# From release root: bash dream_post/rename_release_checkpoints.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${ROOT}/checkpoints"

flatten_recipe() {
  local recipe="$1"
  local dst="${CKPT}/${recipe}"
  mkdir -p "${dst}"

  if [[ -f "${dst}/adapter_config.json" ]]; then
    echo "SKIP (already flat): ${recipe}/"
    return 0
  fi

  local epoch_dir=""
  for d in "${dst}"/epoch_*; do
    [[ -d "$d" ]] || continue
    epoch_dir="$d"
    break
  done

  if [[ -n "${epoch_dir}" ]]; then
    shopt -s dotglob
    mv "${epoch_dir}"/* "${dst}/"
    shopt -u dotglob
    rmdir "${epoch_dir}"
    echo "OK: ${recipe}/ <- $(basename "${epoch_dir}")/"
    return 0
  fi

  echo "WARN: no ${recipe}/ or epoch_* subdir, skipping" >&2
}

migrate_legacy() {
  local src_rel="$1"
  local recipe="$2"
  local epoch_name="$3"
  local src="${CKPT}/${src_rel}"
  local dst="${CKPT}/${recipe}"
  if [[ ! -d "${src}" ]]; then
    return 0
  fi
  mkdir -p "${dst}"
  if [[ -d "${dst}/${epoch_name}" ]]; then
    rm -rf "${src}"
    return 0
  fi
  mv "${src}" "${dst}/${epoch_name}"
  echo "OK: ${src_rel} -> ${recipe}/${epoch_name}"
  find "${CKPT}" -depth -type d -empty -delete 2>/dev/null || true
}

migrate_legacy \
  "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu/w32_a06_lam1_m01_mr08_tdw32_20260424_175927/epoch_1" \
  "dream_prm12k" "epoch_1"
migrate_legacy \
  "ling_coder_SD_response_TD_non_ltr_cut_entropy_rank_grid_w32_bs8_gpu8_allmeth_lr1e4/w32_a08_lam2_m01_mr08_nlcut_tdw32_20260426_100959/epoch_3" \
  "dream_ling_coder" "epoch_3"
migrate_legacy \
  "prm12k_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada/w32_a06_lam1_m02_mr08_tdw32_20260428_012243/epoch_5" \
  "llada_prm12k" "epoch_5"
migrate_legacy \
  "ling_coder_SD_response_TD_random_entropy_rank_grid_bs8_4gpu_llada_default_method_only/w32_a06_lam1_m03_mr08_tdw32_datadef_20260504_235647/epoch_1" \
  "llada_ling_coder" "epoch_1"

for r in dream_prm12k dream_ling_coder llada_prm12k llada_ling_coder; do
  flatten_recipe "$r"
done

echo ""
echo "checkpoints/:"
ls -1 "${CKPT}"

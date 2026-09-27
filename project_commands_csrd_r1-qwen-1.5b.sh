#!/usr/bin/env bash
# CSRD driver (read mode) -- DeepSeek-R1-Distill-Qwen-1.5B student, Qwen3-8B teacher, s1K-1.1. The SFT control is
# the SGL vanilla run of the same config (not retrained here). Each variant is trained, then evaluated (P-ALIGN 4k).
# Usage: GPUS="0" bash project_commands_csrd_r1-qwen-1.5b.sh        (one GPU: variants run one after another)
# New server: scripts/setup.sh (needs PyPI) -> scripts/data/download_r1-qwen-1.5b.sh (needs HF Hub) -> this (offline).
# VARIANTS: space-separated name:lambda:mass_ratio:bands:d_min, in priority order (a finished variant is skipped).
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"
export CSRD_GRAD_LOG_INTERVAL="${CSRD_GRAD_LOG_INTERVAL:-10}"
# Default: lambda 0.1; mass_ratio 1.0 and/or middle band only (both target the far-mass overshoot seen at lambda 0.3).
# Also supported, e.g.: dmin16:0.1:0.1:0,1:16 (builds a second teacher bank).
VARIANTS="${VARIANTS:-main:0.1:0.1:0,1:4 mass1:0.1:1.0:0,1:4 band1:0.1:0.1:0:4 band1-mass1:0.1:1.0:0:4}"

tag_of() {  # tag_of LAMBDA MASS_RATIO BANDS D_MIN -- same rule as scripts/csrd/csrd_lora_r1-qwen-1.5b.sh
  local tag="csrd-lora-l$1"
  [[ "$2" != 0.1 ]] && tag+="-m$2"
  [[ "$3" == 0 ]] && tag+="-b1"
  [[ "$3" == 1 ]] && tag+="-b2"
  [[ "$4" != 4 ]] && tag+="-d$4"
  echo "${tag}-r1-qwen-1.5b"
}

# ============================ DATA ============================
[[ -s data/records/s1k11-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl ]] || bash scripts/data/data_r1-qwen-1.5b.sh
# One teacher bank per d_min the variants use (heads selected once, at d_min 4).
for d in $(for v in ${VARIANTS}; do echo "${v##*:}"; done | sort -un); do
  [[ -s signals/q8b-s1k11-dmin${d}-excess_bg.safetensors ]] || D_MIN=${d} bash scripts/teacher/teacher_qwen3-8b.sh
done

# ======================= TRAIN + EVAL (per variant) ======================
for v in ${VARIANTS}; do
  IFS=: read -r name lambda mass bands dmin <<< "${v}"
  TAG="$(tag_of "${lambda}" "${mass}" "${bands}" "${dmin}")"
  echo "[$(date +%H:%M)] ===== ${name}: ${TAG}"
  CSRD_LAMBDA=${lambda} CSRD_MASS_RATIO=${mass} CSRD_BANDS=${bands} CSRD_D_MIN=${dmin} \
    bash scripts/csrd/csrd_lora_r1-qwen-1.5b.sh
  [[ -f "results-palign/${TAG}/summary.json" ]] || \
    GPUS="${GPUS%% *}" bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/${TAG}" "${TAG}"
done

# =========================== COMPARE ==========================
# Copy SpectralGuidedLearning/results/{vanilla,spectral-lora}-r1-qwen-1.5b into results-palign/ first.
PYTHONPATH="${BASE}/src" "${PROJECT_ENV:-/mnt/local/uvenvs/crsd}/bin/python" "${BASE}/src/compare_results.py" \
  --results-dir results-palign --track r1-qwen-1.5b --baseline vanilla-r1-qwen-1.5b || true

#!/usr/bin/env bash
# CSRD driver (read mode) -- DeepSeek-R1-Distill-Qwen-1.5B student (= the SGL baselines' student), Qwen3-8B teacher
# reading s1K-1.1 (the baselines' data, no regeneration):
#   data (canonical + teacher/student records) -> teacher signal bank (Qwen3-8B, skipped when the bank exists)
#   -> CSRD lambda 0.1 (LoRA, DeepSpeed ZeRO-2, SGL config) -> P-ALIGN eval -> compare.
# The SFT control is the SGL run of the same config (SpectralGuidedLearning, vanilla arm): not retrained here.
#   GPUS="0 1 2 3 4 5 6 7" bash project_commands_csrd_r1-qwen-1.5b.sh
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# Which GPU(s) each phase runs on (space-separated ids). Training uses the whole list (effective batch 32 =
# #GPUs x gradient accumulation), the teacher runs one shard per GPU, eval takes the first id.
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"
export CSRD_LAMBDA="${CSRD_LAMBDA:-0.1}"
TAG="csrd-lora-l${CSRD_LAMBDA}-r1-qwen-1.5b"

# ============================ DATA ============================
[[ -s data/records/s1k11-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl ]] || bash scripts/data/data_r1-qwen-1.5b.sh
# The bank is student-independent (~260 MB): copy signals/q8b-s1k11-dmin4-excess_bg.safetensors from the box that
# built it to skip this phase, or let it rebuild (Qwen3-8B, one shard per GPU).
[[ -s signals/q8b-s1k11-dmin4-excess_bg.safetensors ]] || bash scripts/teacher/teacher_qwen3-8b.sh

# ============================ TRAIN ============================
bash scripts/csrd/csrd_lora_r1-qwen-1.5b.sh

# ============================ EVAL =============================
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
GPUS="${GPUS%% *}" bash scripts/eval/eval_r1-qwen-1.5b.sh "checkpoints/${TAG}" "${TAG}"

# =========================== COMPARE ==========================
# Copy the SGL runs' eval dirs (SpectralGuidedLearning/results/{vanilla,spectral-lora}-r1-qwen-1.5b: same protocol,
# grader and summary/raw format) into results-palign/ first; CSRD is then tested against the vanilla SFT arm.
# Writes results-palign/comparison-table-r1-qwen-1.5b.md (+ gates-g5 / significance json)
PYTHONPATH="${BASE}/src" "${PROJECT_ENV:-/mnt/local/uvenvs/crsd}/bin/python" "${BASE}/src/compare_results.py" \
  --results-dir results-palign --track r1-qwen-1.5b --baseline vanilla-r1-qwen-1.5b || true

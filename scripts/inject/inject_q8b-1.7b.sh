#!/usr/bin/env bash
# Error injection (Sec. 6.7, Appendix C) for one student: ~100 cases (34 per first-reuse bucket [4,16), [16,64), [64,inf)) built
# once from held-out teacher traces, each with a clean control; 4 continuations per case, up to 16k tokens.
# Usage: scripts/inject/inject_q8b-1.7b.sh CKPT TAG
set -euo pipefail

CKPT="${1:?checkpoint dir}"
TAG="${2:?tag}"
read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs "results/inject-${TAG}"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
BASE_MODEL="${LOCAL_MODELS_ROOT}/Qwen3-1.7B-Base"
CASES="data/q8b/inject-cases.jsonl"

[[ -s "${CASES}" ]] || python src/error_injection.py --stage build --records data/q8b/heldout-teacher.jsonl \
  --cases "${CASES}" --per-distance 34 --seed 0
MODEL_OPTS="--model ${CKPT}"
if [[ -f "${CKPT}/adapter_config.json" ]]; then
  LORA_R=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['r'])" "${CKPT}/adapter_config.json")
  MODEL_OPTS+=" --base-model ${BASE_MODEL} --lora-adapter --lora-r ${LORA_R}"
fi
python src/error_injection.py --stage generate --cases "${CASES}" --output "results/inject-${TAG}/continuations.jsonl" \
  ${MODEL_OPTS} --n-samples 4 --max-tokens 16384 2>&1 | tee "logs/inject-${TAG}.log"
python src/error_injection.py --stage score --cases "results/inject-${TAG}/continuations.jsonl" \
  --output "results/inject-${TAG}/report.json" 2>&1 | tee -a "logs/inject-${TAG}.log"

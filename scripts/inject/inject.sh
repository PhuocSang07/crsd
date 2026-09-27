#!/usr/bin/env bash
# Error injection for one student checkpoint: ~100 held-out cases with clean controls (built once per track),
# 4 continuations per case, up to 16k tokens.
# Usage: scripts/inject/inject.sh TRACK CKPT TAG
set -euo pipefail
CKPT="${2:?checkpoint dir}"
TAG="${3:?tag}"
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
CASES="data/inject/${HELDOUT_CANON}${SEG_TAG}-${STUDENT_DIR}-cases.jsonl"
mkdir -p data/inject "results/inject-${TAG}"
[[ -s "${CASES}" ]] || python src/error_injection.py --stage build --records "${STUDENT_HELDOUT_RECORDS}" \
  --cases "${CASES}" --per-distance 34 --seed 0
MODEL_OPTS="--model ${CKPT}"
if [[ -f "${CKPT}/adapter_config.json" ]]; then
  LORA_R=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['r'])" "${CKPT}/adapter_config.json")
  MODEL_OPTS+=" --base-model ${STUDENT} --lora-adapter --lora-r ${LORA_R}"
fi
python src/error_injection.py --stage generate --cases "${CASES}" --output "results/inject-${TAG}/continuations.jsonl" \
  ${MODEL_OPTS} --n-samples 4 --max-tokens 16384 --max-model-len 32768 2>&1 | tee "logs/inject-${TAG}.log"
python src/error_injection.py --stage score --cases "results/inject-${TAG}/continuations.jsonl" \
  --output "results/inject-${TAG}/report.json" 2>&1 | tee -a "logs/inject-${TAG}.log"

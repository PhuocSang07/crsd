#!/usr/bin/env bash
# Evaluate one checkpoint of a track's student (CSRD arms or baseline checkpoints) with one grader, two protocols:
# proposal (n = 16 AIME/AMC, 32k context) -> results-proposal/<TAG>, palign (the baselines' eval) -> results-palign/<TAG>.
# Usage: scripts/eval/eval.sh TRACK CKPT TAG        (CKPT = adapter dir, full model dir, or "base")
# Env: GPU_MEM_UTIL (0.9), PROTOCOLS, PROMPT_STYLE (sgl|zeroshot|fewshot), N_SAMPLES_MAP, DEV_ROLLOUTS=1 (D3), FORCE=1.
set -euo pipefail
MODEL_ARG="${2:?checkpoint dir or 'base'}"
TAG="${3:?tag}"
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
MODEL="${MODEL_ARG}"; [[ "${MODEL_ARG}" == base ]] && MODEL="${STUDENT}"

OPTS=" --model ${MODEL} --tag ${TAG} --benchmarks ${BENCHMARKS:-aime24,aime25,amc12,math500}"
# tensor parallel must divide the student's attention and KV head counts
OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL:-${#GPUS[@]}} --seed 42 --batch-size 64 --prompt-style ${PROMPT_STYLE:-sgl}"
OPTS+=" --template-tokenizer ${STUDENT} --grader palign"
OPTS+=" --gpu-memory-utilization ${GPU_MEM_UTIL:-0.9}"
if [[ -f "${MODEL}/adapter_config.json" ]]; then
  LORA_R=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['r'])" "${MODEL}/adapter_config.json")
  OPTS+=" --base-model ${STUDENT} --lora-adapter --lora-r ${LORA_R}"
fi
for protocol in ${PROTOCOLS:-proposal palign}; do
  RUN_OPTS="${OPTS} --protocol ${protocol}"
  [[ "${protocol}" == proposal && -n "${N_SAMPLES_MAP:-}" ]] && RUN_OPTS+=" --n-samples-map ${N_SAMPLES_MAP}"
  if [[ -f "results-${protocol}/${TAG}/summary.json" && "${FORCE:-0}" != 1 ]]; then
    echo "skip ${TAG} [${protocol}]: results-${protocol}/${TAG}/summary.json exists (FORCE=1 regenerates)"
    continue
  fi
  echo "python src/evaluate.py ${RUN_OPTS}"
  python src/evaluate.py ${RUN_OPTS} 2>&1 | tee "logs/eval-${protocol}-${TAG}.log"
done
if [[ "${DEV_ROLLOUTS:-0}" == 1 && ! -f "results-proposal/${TAG}/dev-rollouts.jsonl" ]]; then
  python src/evaluate.py ${OPTS/--benchmarks ${BENCHMARKS:-aime24,aime25,amc12,math500}/--benchmarks dev} --protocol proposal \
    --n-samples-map dev=4 --tag "${TAG}-dev" --export-traces "results-proposal/${TAG}/dev-rollouts.jsonl" \
    2>&1 | tee "logs/eval-dev-${TAG}.log"
fi

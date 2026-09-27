#!/usr/bin/env bash
# Evaluate one checkpoint of a track's student -- CSRD arms and the SGL / P-ALIGN / SSFT baseline checkpoints alike --
# under both protocols, with one grader (math_verify OR oat_math_grader) and the students' prompt format (SGL):
#   proposal  T 0.6, top-p 0.95, top-k 20, n = 16 AIME/AMC, 4 MATH500, 32k context   -> results-proposal/<TAG>
#   palign    T 0.6, top-p 0.9, rep 1.05, n = 3, 4096 context (the baselines' eval)   -> results-palign/<TAG>
# Usage: scripts/eval/eval.sh TRACK CKPT TAG        (CKPT = adapter dir, full model dir, or "base")
# Env: GPU_MEM_UTIL (0.9), PROTOCOLS ("proposal palign"), PROMPT_STYLE (sgl; zeroshot/fewshot for B0), N_SAMPLES_MAP (pilot:
#      aime24=8,aime25=8,amc12=8), DEV_ROLLOUTS=1 (4 rollouts on the dev set, for D3), FORCE=1 (regenerate).
set -euo pipefail
MODEL_ARG="${2:?checkpoint dir or 'base'}"
TAG="${3:?tag}"
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
MODEL="${MODEL_ARG}"; [[ "${MODEL_ARG}" == base ]] && MODEL="${STUDENT}"

OPTS=" --model ${MODEL} --tag ${TAG} --benchmarks ${BENCHMARKS:-aime24,aime25,amc12,math500}"
# tensor parallel must divide the student's attention and KV head counts (16/8 for 1.7B, 32/8 for 8B): 1, 2, 4 or 8
OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL:-${#GPUS[@]}} --seed 42 --batch-size 64 --prompt-style ${PROMPT_STYLE:-sgl}"
OPTS+=" --template-tokenizer ${STUDENT} --grader palign"
# GPU_MEM_UTIL: vLLM's share of the GPU (default 0.9); lower it when other jobs share the GPU
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

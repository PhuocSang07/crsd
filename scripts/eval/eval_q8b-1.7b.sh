#!/usr/bin/env bash
# Eval a Qwen3-1.7B-Base student with vLLM (Sec. 6.4): AIME24/AIME25/AMC12 at n=16, MATH500 at n=4,
# unbiased pass@1 and pass@3, T=0.6 / top-p 0.95 / top-k 20, 32k context, chat template in thinking mode.
#   ./scripts/eval/eval_q8b-1.7b.sh checkpoints/<tag> <tag>
#   ./scripts/eval/eval_q8b-1.7b.sh base base-zeroshot     (B0; PROMPT_STYLE=zeroshot|fewshot|chat)
# Pilot week 2: N_SAMPLES_MAP=aime24=8,aime25=8,amc12=8 (Table 6); DEV_ROLLOUTS=1 also writes 4 dev rollouts for D3.
set -euo pipefail

MODEL_ARG="${1:?checkpoint dir or 'base'}"
TAG="${2:?tag}"
read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-/mnt/local/_data/aiskylimit_new_nothingnew_2}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ "${VIRTUAL_ENV:-}" != "${PROJECT_ENV}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
BASE_MODEL="${LOCAL_MODELS_ROOT}/Qwen3-1.7B-Base"
TEACHER="${LOCAL_MODELS_ROOT}/Qwen3-8B"
MODEL="${MODEL_ARG}"; [[ "${MODEL_ARG}" == base ]] && MODEL="${BASE_MODEL}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25,amc12,math500}"
PROMPT_STYLE="${PROMPT_STYLE:-chat}"
# Qwen3-1.7B-Base has max_position_embeddings = 32768: prompt + generation must fit in it.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-$(( MAX_MODEL_LEN - 1024 ))}"
RESULTS_DIR="${RESULTS_DIR:-${BASE_PATH}/results}"

OPTS=""
OPTS+=" --model ${MODEL} --tag ${TAG} --benchmarks ${BENCHMARKS}"
OPTS+=" --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens ${MAX_TOKENS} --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --gpu-memory-utilization 0.9 --tensor-parallel-size ${#GPUS[@]} --seed 42 --batch-size 64"
OPTS+=" --prompt-style ${PROMPT_STYLE} --template-tokenizer ${TEACHER} --results-dir ${RESULTS_DIR}"
[[ -n "${N_SAMPLES:-}" ]] && OPTS+=" --n-samples ${N_SAMPLES}"
[[ -n "${N_SAMPLES_MAP:-}" ]] && OPTS+=" --n-samples-map ${N_SAMPLES_MAP}"
if [[ -f "${MODEL}/adapter_config.json" ]]; then
  # vLLM's max_lora_rank must cover the adapter's rank (A13 trains r = 16 / 128 too)
  LORA_R=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['r'])" "${MODEL}/adapter_config.json")
  OPTS+=" --base-model ${BASE_MODEL} --lora-adapter --lora-r ${LORA_R}"
fi

CMD="python ${BASE_PATH}/src/evaluate.py ${OPTS}"
if [[ -f "${RESULTS_DIR}/${TAG}/summary.json" && "${FORCE:-0}" != 1 ]]; then
  echo "skip ${TAG}: ${RESULTS_DIR}/${TAG}/summary.json exists (FORCE=1 regenerates)"
else
  echo "${CMD}"
  ${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"
fi

if [[ "${DEV_ROLLOUTS:-0}" == 1 ]]; then
  DEV_OPTS="${OPTS/--benchmarks ${BENCHMARKS}/--benchmarks dev} --n-samples 4 --n-samples-map dev=4 --tag ${TAG}-dev"
  DEV_OPTS+=" --export-traces ${RESULTS_DIR}/${TAG}/dev-rollouts.jsonl"
  python "${BASE_PATH}/src/evaluate.py" ${DEV_OPTS} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}-dev.log"
fi

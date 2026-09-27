#!/usr/bin/env bash
# Phase 4: eval -- DeepSeek-R1-Distill-Qwen-1.5B track, P-ALIGN protocol and grader (same as SGL eval_r1-qwen-1.5b.sh).
# Usage: scripts/eval/eval_r1-qwen-1.5b.sh [CKPT] [TAG]      (CKPT = adapter dir, full model dir, or "base")
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
# Offline server: models/data come from the download.txt mirrors; no HF Hub access, no vLLM usage stats.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-/mnt/local/_data/aiskylimit_new_nothingnew_2}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: env not found at ${PROJECT_ENV}; build it from crsd.txt (repo root) or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL="${BASE_PATH}/checkpoints/csrd-lora-l0.1-r1-qwen-1.5b"
BASE_MODEL="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
TAG="csrd-lora-l0.1-r1-qwen-1.5b"
[[ -n "${1:-}" ]] && MODEL="$1"
[[ -n "${2:-}" ]] && TAG="$2"
[[ "${MODEL}" == base ]] && MODEL="${BASE_MODEL}"
[[ -d "${BASE_MODEL}" ]] || { echo "missing local model ${BASE_MODEL} (download.txt)" >&2; exit 1; }
[[ -d "${MODEL}" ]] || { echo "missing checkpoint ${MODEL}" >&2; exit 1; }

BENCHMARKS="math500,aime24,aime25,amc12"
PROTOCOL=palign
TEMPERATURE=0.6
TOP_P=0.9
REP_PENALTY=1.05
N_SAMPLES=3
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-$(( MAX_MODEL_LEN - 512 ))}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
# R1-Distill-Qwen-1.5B has 2 KV heads: tensor parallel 1 or 2
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
SEED=42
PROMPT_STYLE=sgl
ENFORCE_EAGER=true
LORA_R=16
RESULTS_DIR="${RESULTS_DIR:-${BASE_PATH}/results-palign}"

OPTS=""
OPTS+=" --model ${MODEL}"
OPTS+=" --tag ${TAG}"
OPTS+=" --protocol ${PROTOCOL}"
OPTS+=" --benchmarks ${BENCHMARKS}"
OPTS+=" --temperature ${TEMPERATURE}"
OPTS+=" --top-p ${TOP_P}"
OPTS+=" --repetition-penalty ${REP_PENALTY}"
OPTS+=" --n-samples ${N_SAMPLES}"
OPTS+=" --max-tokens ${MAX_TOKENS}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --max-model-len ${MAX_MODEL_LEN}"
OPTS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL}"
[[ "${ENFORCE_EAGER}" == true ]] && OPTS+=" --enforce-eager" || OPTS+=" --no-enforce-eager"
OPTS+=" --seed ${SEED}"
OPTS+=" --prompt-style ${PROMPT_STYLE}"
OPTS+=" --grader palign"
OPTS+=" --base-model ${BASE_MODEL}"
if [[ -f "${MODEL}/adapter_config.json" ]]; then
  OPTS+=" --lora-adapter"
  OPTS+=" --lora-r ${LORA_R}"
else
  OPTS+=" --no-lora-adapter"
fi
OPTS+=" --results-dir ${RESULTS_DIR}"

CMD="python ${BASE_PATH}/src/evaluate.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/eval-${TAG}.log"

#!/usr/bin/env bash
# s1K-Q8B (Sec. 6.2): Qwen3-8B (thinking) writes 8 traces per s1K question, one vLLM process per GPU;
# then keep closed, untruncated, correct traces (math-verify, else Qwen3-8B as LLM judge) and pick one
# per question. Writes data/q8b/s1k-traces.jsonl (+ .stats.json: retention, gate G0 >= 600).
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0}"
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
mkdir -p logs data/q8b

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
TEACHER="${LOCAL_MODELS_ROOT}/Qwen3-8B"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K}"
RAW_DIR="data/q8b/s1k-raw"
N_PER_QUESTION=8
MAX_TOKENS=32768
LIMIT="${LIMIT:-}"          # e.g. LIMIT=50 for a dry run

OPTS=""
OPTS+=" --stage generate --source s1k --dataset-name ${DATASET_NAME} --model-name ${TEACHER}"
OPTS+=" --n-per-question ${N_PER_QUESTION} --temperature 0.6 --top-p 0.95 --top-k 20"
OPTS+=" --max-tokens ${MAX_TOKENS} --max-model-len $(( MAX_TOKENS + 2048 )) --seed 42"
[[ -n "${LIMIT}" ]] && OPTS+=" --limit ${LIMIT}"

mkdir -p "${RAW_DIR}"
NUM_SHARDS=${#GPUS[@]}
pids=()
for i in "${!GPUS[@]}"; do
  out="${RAW_DIR}/shard${i}of${NUM_SHARDS}.jsonl"
  if [[ -s "${out}" ]]; then echo "skip ${out} (exists)"; continue; fi
  CMD="python -u ${BASE_PATH}/src/generate_traces.py ${OPTS} --num-shards ${NUM_SHARDS} --shard-index ${i} --output-path ${out}"
  echo "${CMD}"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" ${CMD} > "logs/gen-s1k-q8b-shard${i}.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "${pid}"; done
cat "${RAW_DIR}"/shard*.jsonl > data/q8b/s1k-raw.jsonl

CMD="python ${BASE_PATH}/src/generate_traces.py --stage select --raw-path data/q8b/s1k-raw.jsonl --judge-model ${TEACHER} --output-path data/q8b/s1k-traces.jsonl --seed 42"
echo "${CMD}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" ${CMD} 2>&1 | tee logs/select-s1k-q8b.log
echo ">>> STOP AND READ: G0 needs >= 600 kept questions (data/q8b/s1k-traces.jsonl.stats.json)."

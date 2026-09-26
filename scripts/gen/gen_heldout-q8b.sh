#!/usr/bin/env bash
# Held-out set (Sec. 6.2): 300 correct Qwen3-8B traces on MATH-train levels 3-5, skipping the first 200
# problems (= the dev set, benchmarks.load_dev), after dropping anything sharing a 13-gram with s1K or a
# test set. Used for D1/D2/D4 and error injection.
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
# the 13-gram filter (generate_traces.load_questions) reads s1K and the test sets from here
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
mkdir -p logs data/q8b

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
TEACHER="${LOCAL_MODELS_ROOT}/Qwen3-8B"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/MATH-lighteval}"
DEV_SIZE=200
CANDIDATES=500      # questions sampled; ~300 correct traces kept
KEEP=300

OPTS=""
OPTS+=" --stage generate --source math-train --dataset-name ${DATASET_NAME} --model-name ${TEACHER}"
OPTS+=" --skip ${DEV_SIZE} --limit ${CANDIDATES} --n-per-question 2 --temperature 0.6 --top-p 0.95 --top-k 20"
OPTS+=" --max-tokens 32768 --max-model-len 34816 --seed 42 --output-path data/q8b/heldout-raw.jsonl"
CMD="python ${BASE_PATH}/src/generate_traces.py ${OPTS}"
echo "${CMD}"
[[ -s data/q8b/heldout-raw.jsonl ]] || CUDA_VISIBLE_DEVICES="${GPUS[0]}" ${CMD} 2>&1 | tee logs/gen-heldout-q8b.log

CMD="python ${BASE_PATH}/src/generate_traces.py --stage select --raw-path data/q8b/heldout-raw.jsonl --max-keep ${KEEP} --output-path data/q8b/heldout-traces.jsonl --seed 42"
echo "${CMD}"
${CMD} 2>&1 | tee logs/select-heldout-q8b.log

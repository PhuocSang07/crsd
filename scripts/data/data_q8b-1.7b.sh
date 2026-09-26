#!/usr/bin/env bash
# Nodes + tokens for the Qwen3-8B -> Qwen3-1.7B-Base track: train (s1K-Q8B) and held-out traces, once per
# tokenizer (the two share Qwen3's tokenizer, but the records are built per model on purpose), anchor labels
# (Appendix B) and the 13-gram decontamination report.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs data/q8b

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
TEACHER="${LOCAL_MODELS_ROOT}/Qwen3-8B"
STUDENT="${LOCAL_MODELS_ROOT}/Qwen3-1.7B-Base"
SEGMENT_MODE="${SEGMENT_MODE:-paragraph}"   # A9: paragraph | sentence | episode | chunk3
LABELER="${LABELER:-heuristic}"             # heuristic | llm (Qwen3-8B, the proposal's protocol)
SUFFIX=""; [[ "${SEGMENT_MODE}" != paragraph ]] && SUFFIX="-${SEGMENT_MODE}"

for split in s1k heldout; do
  for role in teacher student; do
    model="${TEACHER}"; [[ "${role}" == student ]] && model="${STUDENT}"
    out="data/q8b/${split}-${role}${SUFFIX}.jsonl"
    OPTS=" --traces-path data/q8b/${split}-traces.jsonl --tokenizer ${model} --output-path ${out}"
    OPTS+=" --segment-mode ${SEGMENT_MODE} --min-step-chars 40 --max-steps 400 --max-tokens 32768"
    CMD="python ${BASE_PATH}/src/data_prep.py ${OPTS}"
    echo "${CMD}"
    ${CMD} 2>&1 | tee "logs/data-${split}-${role}${SUFFIX}.log"
    LABEL_OPTS=" --data-path ${out} --labeler ${LABELER}"
    [[ "${LABELER}" == llm ]] && LABEL_OPTS+=" --model-name ${TEACHER}"
    python "${BASE_PATH}/src/anchor_labels.py" ${LABEL_OPTS} 2>&1 | tee -a "logs/data-${split}-${role}${SUFFIX}.log"
  done
done

python "${BASE_PATH}/src/decontaminate.py" --data data/q8b/s1k-traces.jsonl --data data/q8b/heldout-traces.jsonl \
  --benchmarks aime24,aime25,amc12,math500,dev --output data/q8b/decontamination.json 2>&1 | tee logs/decontaminate-q8b.log

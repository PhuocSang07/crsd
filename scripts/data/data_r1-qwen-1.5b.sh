#!/usr/bin/env bash
# Phase 1: data prep (s1K-1.1, read mode) -- DeepSeek-R1-Distill-Qwen-1.5B track, teacher Qwen3-8B.
#   canonical {question, thinking, answer} from s1K-1.1's DeepSeek-R1 trajectory + attempt (the SGL baselines' data)
#   -> the teacher's rendering (Qwen3-8B, its own thinking template) and the student's training text
#      (R1-Distill-Qwen-1.5B, SGL format: byte-identical to SpectralGuidedLearning/src/data_prep.py)
#   -> both carry the same step nodes (text hashes), which is how the teacher signal bank finds its records.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: env not found at ${PROJECT_ENV}; build it from crsd.txt (repo root) or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
export TOKENIZERS_PARALLELISM=false
# Offline server (network egress is blocked and audited): models/data come from the local mirrors listed in
# download.txt; never contact the HF Hub, and turn off vLLM's usage-stats ping.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
mkdir -p logs data/canonical data/records

# Offline server: no HF Hub access, load from local mirrors (download.txt).
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1}"
TEACHER_NAME="${LOCAL_MODELS_ROOT}/Qwen3-8B"
STUDENT_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
CANONICAL_PATH="data/canonical/s1k11.jsonl"
TEACHER_RECORDS="data/records/s1k11-Qwen3-8B-thinking.jsonl"
STUDENT_RECORDS="data/records/s1k11-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl"
SEGMENT_MODE=paragraph
MIN_STEP_CHARS=40
MAX_STEPS=400
MAX_TOKENS=32768
LABELER=heuristic

for dir in "${TEACHER_NAME}" "${STUDENT_NAME}"; do
  [[ -d "${dir}" ]] || { echo "missing local model ${dir} (download.txt)" >&2; exit 1; }
done
[[ -s "${CANONICAL_PATH}" || -d "${DATASET_NAME}" ]] || { echo "missing local dataset ${DATASET_NAME} (download.txt)" >&2; exit 1; }

# 1. canonical content
if [[ ! -s "${CANONICAL_PATH}" ]]; then
  CMD="python ${BASE_PATH}/src/build_canonical.py --source s1k11 --input ${DATASET_NAME} --output-path ${CANONICAL_PATH}"
  echo "${CMD}"
  ${CMD} 2>&1 | tee logs/r1-qwen-1.5b-canonical.log
fi

# 2. teacher rendering (--style thinking) and student training text (--style sgl), + heuristic anchor labels
for pair in "${TEACHER_NAME}:thinking:${TEACHER_RECORDS}" "${STUDENT_NAME}:sgl:${STUDENT_RECORDS}"; do
  IFS=: read -r TOKENIZER STYLE OUTPUT_PATH <<< "${pair}"
  if [[ -s "${OUTPUT_PATH}" ]] && head -1 "${OUTPUT_PATH}" | grep -q '"anchor"'; then
    echo "skip ${OUTPUT_PATH} (exists)"; continue
  fi
  OPTS=""
  OPTS+=" --canonical ${CANONICAL_PATH}"
  OPTS+=" --tokenizer ${TOKENIZER}"
  OPTS+=" --style ${STYLE}"
  OPTS+=" --segment-mode ${SEGMENT_MODE}"
  OPTS+=" --min-step-chars ${MIN_STEP_CHARS}"
  OPTS+=" --max-steps ${MAX_STEPS}"
  OPTS+=" --max-tokens ${MAX_TOKENS}"
  OPTS+=" --output-path ${OUTPUT_PATH}"
  CMD="python ${BASE_PATH}/src/data_prep.py ${OPTS}"
  echo "${CMD}"
  ${CMD} 2>&1 | tee "logs/r1-qwen-1.5b-records-$(basename "${OUTPUT_PATH}" .jsonl).log"
  python "${BASE_PATH}/src/anchor_labels.py" --data-path "${OUTPUT_PATH}" --labeler "${LABELER}" \
    2>&1 | tee -a "logs/r1-qwen-1.5b-records-$(basename "${OUTPUT_PATH}" .jsonl).log"
done

# 3. the student trains only on traces the teacher has signals for (a trace near 32k tokens can pass the length
#    filter under one tokenizer and not the other): keep the id intersection on the student side.
python - "${TEACHER_RECORDS}" "${STUDENT_RECORDS}" <<'PY'
import json, sys
teacher, student = sys.argv[1:]
ids = {json.loads(line)["id"] for line in open(teacher)}
rows = open(student).readlines()
kept = [row for row in rows if json.loads(row)["id"] in ids]
if len(kept) < len(rows):
    open(student, "w").writelines(kept)
print(f"{student}: {len(kept)}/{len(rows)} records have a teacher rendering")
PY

echo ">>> STOP AND READ: 'wrote 1000 records' for both files above (s1K-1.1 has 1000 traces, all under 32k tokens)."

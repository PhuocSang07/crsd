#!/usr/bin/env bash
# Held-out traces for the MC-CSRD diagnostics (v4 Sec. 7.4) -- 300 OpenR1-Math R1 traces rendered for the Qwen3-8B teacher
# (thinking) and the R1-Distill-Qwen-1.5B student (SGL), with the segmentation of data_r1-qwen-1.5b.sh.
# Needs data/canonical/openr1-heldout.jsonl: copy it from the machine that built it (5 MB, 13-gram-disjoint from s1K and
# the test sets), or mirror open-r1/OpenR1-Math-220k into LOCAL_DATA_ROOT (download.txt) and it is rebuilt here.
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
# Offline server: models/data come from the download.txt mirrors; no HF Hub access.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
mkdir -p logs data/canonical data/records

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
OPENR1_PATH="${OPENR1_PATH:-${LOCAL_DATA_ROOT}/OpenR1-Math-220k}"
TEACHER_NAME="${LOCAL_MODELS_ROOT}/Qwen3-8B"
STUDENT_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
CANONICAL_PATH="data/canonical/openr1-heldout.jsonl"
TEACHER_RECORDS="data/records/openr1-heldout-Qwen3-8B-thinking.jsonl"
STUDENT_RECORDS="data/records/openr1-heldout-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl"
N_HELDOUT=300
SEGMENT_MODE=paragraph
MIN_STEP_CHARS=40
MAX_STEPS=400
MAX_TOKENS=32768
LABELER=heuristic

if [[ ! -s "${CANONICAL_PATH}" ]]; then
  [[ -d "${OPENR1_PATH}" ]] || {
    echo "missing ${CANONICAL_PATH}: copy it from the dev box, or mirror open-r1/OpenR1-Math-220k to ${OPENR1_PATH}" >&2
    exit 1
  }
  CMD="python ${BASE_PATH}/src/build_canonical.py --source openr1 --input ${OPENR1_PATH} --limit ${N_HELDOUT} --seed 42 --output-path ${CANONICAL_PATH}"
  echo "${CMD}"
  ${CMD} 2>&1 | tee logs/r1-qwen-1.5b-canonical-heldout.log
fi

for pair in "${TEACHER_NAME}:thinking:${TEACHER_RECORDS}" "${STUDENT_NAME}:sgl:${STUDENT_RECORDS}"; do
  IFS=: read -r TOKENIZER STYLE OUTPUT_PATH <<< "${pair}"
  if [[ -s "${OUTPUT_PATH}" ]] && head -1 "${OUTPUT_PATH}" | grep -q '"anchor"'; then
    echo "skip ${OUTPUT_PATH} (exists)"; continue
  fi
  [[ -d "${TOKENIZER}" ]] || { echo "missing local model ${TOKENIZER} (download.txt)" >&2; exit 1; }
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

# Near 32k tokens the length filter can differ by tokenizer: keep only student traces the teacher also has.
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

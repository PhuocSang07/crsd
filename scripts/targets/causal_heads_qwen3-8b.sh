#!/usr/bin/env bash
# CSRD-C prerequisites (decision tree: G1 true, G4 false): pick teacher heads by attention-causal agreement.
#   1. per-head routing of every band head on 40 traces of the causal subset (per-head R of all 704 band heads
#      is ~0.2 GB per trace in float16, so the whole subset would be ~45 GB)
#   2. median Spearman(R_head, C~) per head -> top-16 per band -> heads-causal.json
#   3. routing targets of those heads on all train traces -> data/q8b/targets-dmin4-causal
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
mkdir -p logs

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
TEACHER="${LOCAL_MODELS_ROOT}/Qwen3-8B"
D_MIN="${D_MIN:-4}"
ROUTING_DIR="data/q8b/routing-teacher"
CAUSAL="data/q8b/causal"
ALLBAND="data/q8b/causal-subset-allband"

# records of the causal subset only
python - <<PY
import json, os
ids = {f[:-4] for f in os.listdir("${CAUSAL}") if f.endswith(".npz")}
with open("data/q8b/s1k-teacher.jsonl") as src, open("data/q8b/causal-subset-teacher.jsonl", "w") as dst:
    for line in src:
        if json.loads(line)["id"] in ids:
            dst.write(line)
print(f"{len(ids)} causal-subset traces")
PY

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u "${BASE_PATH}/src/extract_routing.py" --stage targets --model-name "${TEACHER}" \
  --data-path data/q8b/causal-subset-teacher.jsonl --heads-json "${ROUTING_DIR}/heads-allband.json" \
  --output-dir "${ALLBAND}" --d-min ${D_MIN} --save-per-head --limit ${CAUSAL_HEAD_TRACES:-40} \
  2>&1 | tee logs/causal-allband-q8b.log
python "${BASE_PATH}/src/diagnostics.py" --teacher-targets "${ALLBAND}" --student-targets "${ALLBAND}" \
  --causal-dir "${CAUSAL}" --output-dir results/diag-teacher-causal-heads --bootstrap 10 \
  --write-causal-heads "${ROUTING_DIR}/heads-causal.json" --num-layers 36 --k-per-band 16
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u "${BASE_PATH}/src/extract_routing.py" --stage targets --model-name "${TEACHER}" \
  --data-path data/q8b/s1k-teacher.jsonl --heads-json "${ROUTING_DIR}/heads-causal.json" \
  --output-dir "data/q8b/targets-dmin${D_MIN}-causal" --d-min ${D_MIN} 2>&1 | tee logs/targets-causal-q8b.log

#!/usr/bin/env bash
# CSRD-C prerequisites (decision tree: G1 true, G4 false): teacher heads chosen by attention-causal agreement.
#   1. per-head routing of every band head on 40 traces of the causal subset (all band heads' per-head R in
#      float16 is ~0.2 GB per trace for the 8B teacher, so the whole subset would not fit comfortably)
#   2. median Spearman(R_head, C~) per head -> top-16 per band -> heads-causal.json
#   3. routing targets of those heads on every train trace, packed with the causal targets into
#      signals/<TT>-<CANON>-dmin<D>-causal.safetensors (used by the csrd-c and csrd-causalonly arms)
# Usage: scripts/targets/causal_heads.sh TRACK
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
DEVICE_OPTS=(); [[ "${TEACHER_GPUS}" -gt 1 ]] && DEVICE_OPTS=(--device-map auto)
ROUTING="${TEACHER_WORK}/routing"
CAUSAL="${TEACHER_WORK}/causal-dmin${D_MIN}-${SCORE}"
ALLBAND="${TEACHER_WORK}/causal-subset-allband"
SUBSET="${TEACHER_WORK}/causal-subset-records.jsonl"
NUM_LAYERS=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['num_layers'])" "${ROUTING}/heads-${SCORE}.json")

python - "${CAUSAL}" "${TEACHER_TRAIN_RECORDS}" "${SUBSET}" <<'PY'
import json, os, sys
causal, records, subset = sys.argv[1:]
ids = {f[:-4] for f in os.listdir(causal) if f.endswith(".npz")}
with open(records) as src, open(subset, "w") as dst:
    for line in src:
        if json.loads(line)["id"] in ids:
            dst.write(line)
print(f"{len(ids)} causal-subset traces")
PY
CUDA_VISIBLE_DEVICES="${GPU_GROUPS[0]}" python -u src/extract_routing.py --stage targets --model-name "${TEACHER}" \
  --data-path "${SUBSET}" --heads-json "${ROUTING}/heads-allband.json" --output-dir "${ALLBAND}" --d-min ${D_MIN} \
  --save-per-head --limit "${CAUSAL_HEAD_TRACES:-40}" "${DEVICE_OPTS[@]}" 2>&1 | tee "logs/causal-allband-${TT}.log"
python src/diagnostics.py --teacher-targets "${ALLBAND}" --student-targets "${ALLBAND}" --causal-dir "${CAUSAL}" \
  --output-dir "results/diag-teacher-causal-heads-${TT}-${TRAIN_CANON}" --bootstrap 10 \
  --write-causal-heads "${ROUTING}/heads-causal.json" --num-layers "${NUM_LAYERS}" --k-per-band 16
TARGETS="${TEACHER_WORK}/targets-dmin${D_MIN}-causal"
sharded "targets-causal-${TT}" python -u src/extract_routing.py --stage targets --model-name "${TEACHER}" \
  --data-path "${TEACHER_TRAIN_RECORDS}" --heads-json "${ROUTING}/heads-causal.json" --output-dir "${TARGETS}" \
  --d-min ${D_MIN} --source-name "${TRAIN_CANON}" "${DEVICE_OPTS[@]}"
python src/signal_bank.py pack --targets-dir "${TARGETS}" --causal-dir "${CAUSAL}" \
  --output "signals/${TT}-${TRAIN_CANON}${SEG_TAG}-dmin${D_MIN}-causal.safetensors"

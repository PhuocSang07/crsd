#!/usr/bin/env bash
# Records for one track: teacher rendering (thinking) and student SGL text for train and held-out, anchor labels,
# and the 13-gram decontamination report.
# Usage: scripts/data/records.sh TRACK        (SEGMENT_MODE=sentence|episode|chunk3 for ablation A9)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
LABELER="${LABELER:-heuristic}"   # heuristic | llm (Qwen3-8B, the proposal's protocol)

prep() {  # prep CANON MODEL_DIR STYLE
  local out; out="$(records_path "$1" "$2" "$3")"
  # skip only anchor-labelled files, else D2 / CSRD-A would silently see no anchors
  if [[ -s "${out}" ]] && head -1 "${out}" | grep -q '"anchor"'; then echo "skip ${out} (exists)"; return; fi
  python src/data_prep.py --canonical "data/canonical/$1.jsonl" --tokenizer "${LOCAL_MODELS_ROOT}/$2" --style "$3" \
    --segment-mode "${SEGMENT_MODE}" --min-step-chars 40 --max-steps 400 --max-tokens 32768 --output-path "${out}" \
    2>&1 | tee "logs/records-$(basename "${out}" .jsonl).log"
  local label_opts="--data-path ${out} --labeler ${LABELER}"
  [[ "${LABELER}" == llm ]] && label_opts+=" --model-name ${JUDGE}"
  python src/anchor_labels.py ${label_opts} 2>&1 | tee -a "logs/records-$(basename "${out}" .jsonl).log"
}
for canon in "${TRAIN_CANON}" "${HELDOUT_CANON}"; do
  prep "${canon}" "${TEACHER_DIR}" thinking
  prep "${canon}" "${STUDENT_DIR}" sgl
  # Near 32k tokens the length filter can differ by tokenizer: keep only student traces the teacher also has.
  python - "$(records_path "${canon}" "${TEACHER_DIR}" thinking)" "$(records_path "${canon}" "${STUDENT_DIR}" sgl)" <<'PY'
import json, sys
teacher, student = sys.argv[1:]
ids = {json.loads(line)["id"] for line in open(teacher)}
rows = open(student).readlines()
kept = [row for row in rows if json.loads(row)["id"] in ids]
if len(kept) < len(rows):
    open(student, "w").writelines(kept)
print(f"{student}: {len(kept)}/{len(rows)} records have a teacher rendering")
PY
done
python src/decontaminate.py --data "data/canonical/${TRAIN_CANON}.jsonl" --data "data/canonical/${HELDOUT_CANON}.jsonl" \
  --benchmarks aime24,aime25,amc12,math500,dev --output "data/canonical/decontamination-${TRACK}.json" \
  2>&1 | tee "logs/decontaminate-${TRACK}.log"

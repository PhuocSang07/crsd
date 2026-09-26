#!/usr/bin/env bash
# Canonical content (question / thinking / answer) for a track's train and held-out sets.
#   read-*: s1K-1.1 DeepSeek-R1 traces (the baselines' data) + 300 OpenR1-Math R1 traces (held-out)
#   gen-*:  the teacher's own traces from scripts/gen/gen_traces.sh
# Usage: scripts/data/canonical.sh TRACK
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"

build() {  # build CANON args...
  local canon=$1; shift
  [[ -s "data/canonical/${canon}.jsonl" ]] && { echo "skip data/canonical/${canon}.jsonl (exists)"; return; }
  python src/build_canonical.py "$@" --output-path "data/canonical/${canon}.jsonl" 2>&1 | tee "logs/canonical-${canon}.log"
}
local_or_hub() {  # local mirror (download.txt) if present, else the HF Hub id
  [[ -d "${LOCAL_DATA_ROOT}/$1" ]] && echo "${LOCAL_DATA_ROOT}/$1" || echo "$2"
}
if [[ "${MODE}" == read ]]; then
  build "${TRAIN_CANON}" --source s1k11 --input "${S1K11_PATH:-$(local_or_hub s1K-1.1 simplescaling/s1K-1.1)}"
  build "${HELDOUT_CANON}" --source openr1 --input "${OPENR1_PATH:-$(local_or_hub OpenR1-Math-220k open-r1/OpenR1-Math-220k)}" \
    --limit 300 --seed 42
else
  build "${TRAIN_CANON}" --source jsonl --input "data/gen/${TT}/s1k-traces.jsonl"
  build "${HELDOUT_CANON}" --source jsonl --input "data/gen/${TT}/heldout-traces.jsonl"
fi

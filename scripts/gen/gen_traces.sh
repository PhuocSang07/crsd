#!/usr/bin/env bash
# gen-* tracks only: the teacher writes its own traces, one vLLM process per GPU group. s1k: 8 per question, one
# correct kept per question (G0 >= 600); heldout: 300 correct MATH-train traces. Judge: math-verify, else Qwen3-8B.
# Usage: scripts/gen/gen_traces.sh gen-q8b-1.7b|gen-d32b-q8b     (LIMIT=20 for a dry run)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
[[ "${MODE}" == gen ]] || { echo "${TRACK} reads existing traces; nothing to generate" >&2; exit 0; }
OUT="data/gen/${TT}"
mkdir -p "${OUT}"

generate() {  # generate NAME SOURCE extra-opts...
  local name=$1 source=$2; shift 2
  [[ -s "${OUT}/${name}-raw.jsonl" ]] && { echo "skip ${OUT}/${name}-raw.jsonl (exists)"; return; }
  mkdir -p "${OUT}/${name}-raw"
  local opts="--stage generate --source ${source} --model-name ${TEACHER} --tensor-parallel-size ${TEACHER_GPUS}"
  opts+=" --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 32768 --max-model-len 34816 --seed 42"
  sharded "gen-${TT}-${name}" python -u src/generate_traces.py ${opts} "$@" --output-path "${OUT}/${name}-raw/part.jsonl"
  cat "${OUT}/${name}-raw"/part.jsonl* > "${OUT}/${name}-raw.jsonl"
}
LIMIT_OPTS=(); [[ -n "${LIMIT:-}" ]] && LIMIT_OPTS=(--limit "${LIMIT}")
local_or_hub() { [[ -d "${LOCAL_DATA_ROOT}/$1" ]] && echo "${LOCAL_DATA_ROOT}/$1" || echo "$2"; }
generate s1k s1k --dataset-name "$(local_or_hub s1K simplescaling/s1K)" --n-per-question 8 "${LIMIT_OPTS[@]}"
generate heldout math-train --dataset-name "$(local_or_hub MATH-lighteval DigitalLearningGmbH/MATH-lighteval)" \
  --skip 200 --limit "${LIMIT:-500}" --n-per-question 2

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python src/generate_traces.py --stage select --raw-path "${OUT}/s1k-raw.jsonl" \
  --judge-model "${JUDGE}" --output-path "${OUT}/s1k-traces.jsonl" --seed 42 2>&1 | tee "logs/select-${TT}-s1k.log"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python src/generate_traces.py --stage select --raw-path "${OUT}/heldout-raw.jsonl" \
  --max-keep 300 --output-path "${OUT}/heldout-traces.jsonl" --seed 42 2>&1 | tee "logs/select-${TT}-heldout.log"
echo ">>> STOP AND READ: G0 needs >= 600 kept questions (${OUT}/s1k-traces.jsonl.stats.json)."

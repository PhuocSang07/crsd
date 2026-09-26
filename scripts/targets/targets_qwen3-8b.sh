#!/usr/bin/env bash
# Teacher side of CSRD (Sec. 4.3-4.4), Qwen3-8B reading its own traces:
#   1. calibrate receiver scores of all 36x32 heads on D_cal (200 train traces), one shard per GPU
#   2. select top-16 heads per depth band (excess kurtosis after background subtraction; raw kurtosis,
#      random and whole-band selections are written too, for A2/A5)
#   3. routing targets P, Z on every train and held-out trace (per-head R on held-out for D4)
#   4. causal targets by attention suppression on 20% of train traces and 20 held-out traces (D4)
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
SCORE="${SCORE:-excess_bg}"                  # excess_bg | kurtosis | random | allband
ROUTING_DIR="data/q8b/routing-teacher"
TARGETS="data/q8b/targets-dmin${D_MIN}-${SCORE}"
HELDOUT_TARGETS="data/q8b/heldout-targets-dmin${D_MIN}-${SCORE}"
CAUSAL="data/q8b/causal"
HELDOUT_CAUSAL="data/q8b/heldout-causal"
N_CAL=200
K_PER_BAND=16
ATTN=sdpa
NUM_SHARDS=${#GPUS[@]}

sharded() {  # run "$@" once per GPU with --num-shards/--shard-index, wait for all
  local name=$1; shift
  local pids=()
  for i in "${!GPUS[@]}"; do
    echo "$* --num-shards ${NUM_SHARDS} --shard-index ${i}"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$@" --num-shards "${NUM_SHARDS}" --shard-index "${i}" > "logs/${name}-shard${i}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "shard ${i} failed: logs/${name}-shard${i}.log" >&2; fail=1; }; done
  return ${fail}
}

if ! ls "${ROUTING_DIR}"/calib-*.npz >/dev/null 2>&1; then
  sharded calibrate-q8b python -u "${BASE_PATH}/src/extract_routing.py" --stage calibrate --model-name "${TEACHER}" \
    --data-path data/q8b/s1k-teacher.jsonl --output-dir "${ROUTING_DIR}" --n-traces ${N_CAL} --d-min ${D_MIN} \
    --attn-implementation ${ATTN}
fi
python "${BASE_PATH}/src/extract_routing.py" --stage select --output-dir "${ROUTING_DIR}" --k-per-band ${K_PER_BAND} \
  2>&1 | tee logs/select-heads-q8b.log
echo ">>> STOP AND READ: split-half stability of the receiver heads above (Bogdan et al.: r = .67)."

HEADS="${ROUTING_DIR}/heads-${SCORE}.json"
sharded targets-q8b python -u "${BASE_PATH}/src/extract_routing.py" --stage targets --model-name "${TEACHER}" \
  --data-path data/q8b/s1k-teacher.jsonl --heads-json "${HEADS}" --output-dir "${TARGETS}" --d-min ${D_MIN} \
  --attn-implementation ${ATTN}
sharded heldout-targets-q8b python -u "${BASE_PATH}/src/extract_routing.py" --stage targets --model-name "${TEACHER}" \
  --data-path data/q8b/heldout-teacher.jsonl --heads-json "${HEADS}" --output-dir "${HELDOUT_TARGETS}" --d-min ${D_MIN} \
  --attn-implementation ${ATTN} --save-per-head

if [[ "${SKIP_CAUSAL:-false}" != true ]]; then
  sharded causal-q8b python -u "${BASE_PATH}/src/causal_targets.py" --model-name "${TEACHER}" \
    --data-path data/q8b/s1k-teacher.jsonl --targets-dir "${TARGETS}" --output-dir "${CAUSAL}" --fraction 0.2 \
    --top-j 24 --d-min ${D_MIN}
  sharded heldout-causal-q8b python -u "${BASE_PATH}/src/causal_targets.py" --model-name "${TEACHER}" \
    --data-path data/q8b/heldout-teacher.jsonl --targets-dir "${HELDOUT_TARGETS}" --output-dir "${HELDOUT_CAUSAL}" \
    --fraction 1.0 --limit 20 --top-j 24 --d-min ${D_MIN}
fi

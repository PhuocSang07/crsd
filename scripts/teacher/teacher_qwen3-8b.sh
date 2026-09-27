#!/usr/bin/env bash
# Phase 2: teacher signals (read mode) -- Qwen3-8B teacher-forced on s1K-1.1 (its own thinking rendering).
#   calibrate receiver scores of every head on D_cal (first 200 traces) -> top-16 heads per depth band (excess
#   kurtosis after background subtraction) -> far routing targets P, Z on all 1000 traces -> ONE signal bank.
# The bank depends on (teacher, data) only, never on the student: any student whose records carry the same step
# nodes trains from it. A bank copied from another machine is reused as is (delete it to rebuild).
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || ./scripts/setup.sh
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs signals

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen3-8B"
DATA_PATH="data/records/s1k11-Qwen3-8B-thinking.jsonl"
ROUTING_DIR="data/teacher/q8b-s1k11/routing"
TARGETS_DIR="data/teacher/q8b-s1k11/targets-dmin4-excess_bg"
SIGNALS_PATH="signals/q8b-s1k11-dmin4-excess_bg.safetensors"
SOURCE_NAME=s1k11
N_CAL=200
K_PER_BAND=16
D_MIN=4
SCORE=excess_bg
[[ -f "${DATA_PATH}" ]] || { echo "missing ${DATA_PATH}: run scripts/data/data_r1-qwen-1.5b.sh first" >&2; exit 1; }

if [[ -s "${SIGNALS_PATH}" ]]; then
  echo "signal bank ${SIGNALS_PATH} already exists (reused; delete it to rebuild)"
  python "${BASE_PATH}/src/signal_bank.py" info "${SIGNALS_PATH}"
  exit 0
fi

run_shards() {  # run_shards NAME args...: one process per GPU (--num-shards/--shard-index), wait for all
  local name=$1; shift
  local num_shards=${#GPUS[@]} pids=() shard_fail=0
  echo ">>> launching ${num_shards} shards (one per GPU: ${GPUS[*]}) of ${BASE_PATH}/src/extract_routing.py $1 $2"
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python -u "${BASE_PATH}/src/extract_routing.py" "$@" \
      --num-shards "${num_shards}" --shard-index "${i}" > "logs/q8b-s1k11-${name}-shard${i}.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "shard ${i} failed -- see logs/q8b-s1k11-${name}-shard${i}.log" >&2; shard_fail=1; }
  done
  [[ ${shard_fail} -eq 0 ]] || exit 1
}

# 1. calibration, sharded over the traces. Shards from another GPU count would leave D_cal incomplete: redo.
NUM_SHARDS=${#GPUS[@]}
if [[ $(ls "${ROUTING_DIR}"/calib-shard*of${NUM_SHARDS}.npz 2>/dev/null | wc -l) -ne ${NUM_SHARDS} \
      || $(ls "${ROUTING_DIR}"/calib-*.npz 2>/dev/null | wc -l) -ne ${NUM_SHARDS} ]]; then
  rm -f "${ROUTING_DIR}"/calib-*.npz
  OPTS=""
  OPTS+=" --stage calibrate"
  OPTS+=" --model-name ${MODEL_NAME}"
  OPTS+=" --data-path ${DATA_PATH}"
  OPTS+=" --output-dir ${ROUTING_DIR}"
  OPTS+=" --n-traces ${N_CAL}"
  OPTS+=" --d-min ${D_MIN}"
  run_shards calibrate ${OPTS}
fi

# 2. head selection (CPU): merges the calibration shards
OPTS=""
OPTS+=" --stage select"
OPTS+=" --output-dir ${ROUTING_DIR}"
OPTS+=" --k-per-band ${K_PER_BAND}"
OPTS+=" --expected-traces ${N_CAL}"
CMD="python ${BASE_PATH}/src/extract_routing.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/q8b-s1k11-select-heads.log
echo ">>> STOP AND READ ${ROUTING_DIR}/selection-summary.json: split-half stability (this box: 0.998) and score agreement."

# 3. routing targets on every trace (resumable: finished traces are skipped)
OPTS=""
OPTS+=" --stage targets"
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --heads-json ${ROUTING_DIR}/heads-${SCORE}.json"
OPTS+=" --output-dir ${TARGETS_DIR}"
OPTS+=" --d-min ${D_MIN}"
OPTS+=" --source-name ${SOURCE_NAME}"
run_shards targets ${OPTS}

# 4. pack into the reusable bank
CMD="python ${BASE_PATH}/src/signal_bank.py pack --targets-dir ${TARGETS_DIR} --output ${SIGNALS_PATH}"
echo "${CMD}"
${CMD} 2>&1 | tee logs/q8b-s1k11-pack.log

echo ">>> STOP AND READ: 'packed 1000 traces' above (one per s1K-1.1 record)."

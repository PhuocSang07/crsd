#!/usr/bin/env bash
# Teacher side of CSRD (Sec. 4.3-4.4) for one track, teacher-forced on its own rendering (--style thinking):
#   1. calibrate receiver scores of every head on D_cal (first 200 train traces)
#   2. select top-16 heads per depth band (excess kurtosis after background subtraction; raw kurtosis, random
#      and whole-band selections written too, for A2/A5)
#   3. routing targets P, Z on every train trace; causal targets by attention suppression on 20% of them
#   4. pack both into ONE reusable signal bank, keyed by node-text hashes: signals/<TT>-<CANON>-dmin<D>-<SCORE>.safetensors
#      -- any student (any tokenizer) whose records have the same nodes trains from it, no teacher rerun
#   5. held-out traces: targets with per-head R (D4) and causal targets on 20 traces, kept as a directory
# Every stage resumes (skips finished traces / outputs). The 32B teacher runs sharded over TEACHER_GPUS GPUs.
# Usage: scripts/targets/teacher_signals.sh TRACK
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
N_CAL=200
K_PER_BAND=16
DEVICE_OPTS=(); [[ "${TEACHER_GPUS}" -gt 1 ]] && DEVICE_OPTS=(--device-map auto)
ROUTING="${TEACHER_WORK}/routing"
TARGETS="${TEACHER_WORK}/targets-dmin${D_MIN}-${SCORE}"
CAUSAL="${TEACHER_WORK}/causal-dmin${D_MIN}-${SCORE}"

N_RECORDS=$(wc -l < "${TEACHER_TRAIN_RECORDS}")
N_EXPECT=$(( N_RECORDS < N_CAL ? N_RECORDS : N_CAL ))
calibration_complete() {  # every shard of the current GPU grouping is present
  local n=${#GPU_GROUPS[@]}
  for (( i = 0; i < n; i++ )); do [[ -f "${ROUTING}/calib-shard${i}of${n}.npz" ]] || return 1; done
  [[ $(ls "${ROUTING}"/calib-*.npz | wc -l) -eq ${n} ]]
}

if [[ -s "${SIGNALS}" ]]; then
  echo "signal bank ${SIGNALS} already exists (reused; delete it to rebuild)"
  # a bank copied from another machine carries its head selection; the held-out stage below needs it
  [[ -f "${ROUTING}/heads-${SCORE}.json" ]] || python src/signal_bank.py heads "${SIGNALS}" "${ROUTING}/heads-${SCORE}.json"
else
  if ! calibration_complete; then
    rm -f "${ROUTING}"/calib-*.npz  # partial or from another GPU count: redo, or D_cal would silently shrink
    sharded "calibrate-${TT}-${TRAIN_CANON}" python -u src/extract_routing.py --stage calibrate --model-name "${TEACHER}" \
      --data-path "${TEACHER_TRAIN_RECORDS}" --output-dir "${ROUTING}" --n-traces ${N_CAL} --d-min ${D_MIN} "${DEVICE_OPTS[@]}"
  fi
  python src/extract_routing.py --stage select --output-dir "${ROUTING}" --k-per-band ${K_PER_BAND} \
    --expected-traces ${N_EXPECT} 2>&1 | tee "logs/select-heads-${TT}-${TRAIN_CANON}.log"
  echo ">>> STOP AND READ ${ROUTING}/selection-summary.json: split-half stability (Bogdan et al.: r = .67) and score agreement."
  sharded "targets-${TT}-${TRAIN_CANON}" python -u src/extract_routing.py --stage targets --model-name "${TEACHER}" \
    --data-path "${TEACHER_TRAIN_RECORDS}" --heads-json "${ROUTING}/heads-${SCORE}.json" --output-dir "${TARGETS}" \
    --d-min ${D_MIN} --source-name "${TRAIN_CANON}" "${DEVICE_OPTS[@]}"
  if [[ "${SKIP_CAUSAL:-false}" != true ]]; then
    sharded "causal-${TT}-${TRAIN_CANON}" python -u src/causal_targets.py --model-name "${TEACHER}" \
      --data-path "${TEACHER_TRAIN_RECORDS}" --targets-dir "${TARGETS}" --output-dir "${CAUSAL}" --fraction 0.2 \
      --top-j 24 --d-min ${D_MIN} "${DEVICE_OPTS[@]}"
    grep -h "WARNING" logs/causal-${TT}-${TRAIN_CANON}-shard*.log || true
  fi
  CAUSAL_OPTS=(); [[ -d "${CAUSAL}" ]] && CAUSAL_OPTS=(--causal-dir "${CAUSAL}")
  python src/signal_bank.py pack --targets-dir "${TARGETS}" "${CAUSAL_OPTS[@]}" --output "${SIGNALS}" \
    2>&1 | tee "logs/pack-$(basename "${SIGNALS}" .safetensors).log"
fi

HO_TARGETS="${TEACHER_HELDOUT_WORK}/targets-dmin${D_MIN}-${SCORE}"
if [[ $(ls "${HO_TARGETS}"/*.npz 2>/dev/null | wc -l) -lt $(wc -l < "${TEACHER_HELDOUT_RECORDS}") ]]; then
  sharded "heldout-targets-${TT}-${HELDOUT_CANON}" python -u src/extract_routing.py --stage targets --model-name "${TEACHER}" \
    --data-path "${TEACHER_HELDOUT_RECORDS}" --heads-json "${ROUTING}/heads-${SCORE}.json" --output-dir "${HO_TARGETS}" \
    --d-min ${D_MIN} --save-per-head --source-name "${HELDOUT_CANON}" "${DEVICE_OPTS[@]}"
fi
if [[ "${SKIP_CAUSAL:-false}" != true && $(ls "${TEACHER_HELDOUT_WORK}/causal-dmin${D_MIN}-${SCORE}"/*.npz 2>/dev/null | wc -l) -lt 20 ]]; then
  sharded "heldout-causal-${TT}-${HELDOUT_CANON}" python -u src/causal_targets.py --model-name "${TEACHER}" \
    --data-path "${TEACHER_HELDOUT_RECORDS}" --targets-dir "${HO_TARGETS}" --output-dir "${TEACHER_HELDOUT_WORK}/causal-dmin${D_MIN}-${SCORE}" \
    --fraction 1.0 --limit 20 --top-j 24 --d-min ${D_MIN} "${DEVICE_OPTS[@]}"
fi
python src/signal_bank.py info "${SIGNALS}"

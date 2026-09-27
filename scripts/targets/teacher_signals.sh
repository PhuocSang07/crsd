#!/usr/bin/env bash
# Teacher signals for one track: head calibration on D_cal (first 200 traces) -> top-16 heads per band -> routing
# targets (+ causal targets on 20%) -> one reusable signal bank; then held-out targets. Every stage resumes.
# Env: SKIP_TRAIN_CAUSAL=true skips train-side causal targets (only CSRD-C needs them); SKIP_CAUSAL=true skips all.
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
  # a copied bank carries its head selection, which the held-out stage needs
  [[ -f "${ROUTING}/heads-${SCORE}.json" ]] || python src/signal_bank.py heads "${SIGNALS}" "${ROUTING}/heads-${SCORE}.json"
else
  if ! calibration_complete; then
    rm -f "${ROUTING}"/calib-*.npz  # stale shards would silently shrink D_cal
    sharded "calibrate-${TT}-${TRAIN_CANON}" python -u src/extract_routing.py --stage calibrate --model-name "${TEACHER}" \
      --data-path "${TEACHER_TRAIN_RECORDS}" --output-dir "${ROUTING}" --n-traces ${N_CAL} --d-min ${D_MIN} "${DEVICE_OPTS[@]}"
  fi
  python src/extract_routing.py --stage select --output-dir "${ROUTING}" --k-per-band ${K_PER_BAND} \
    --expected-traces ${N_EXPECT} 2>&1 | tee "logs/select-heads-${TT}-${TRAIN_CANON}.log"
  echo ">>> STOP AND READ ${ROUTING}/selection-summary.json: split-half stability (Bogdan et al.: r = .67) and score agreement."
  sharded "targets-${TT}-${TRAIN_CANON}" python -u src/extract_routing.py --stage targets --model-name "${TEACHER}" \
    --data-path "${TEACHER_TRAIN_RECORDS}" --heads-json "${ROUTING}/heads-${SCORE}.json" --output-dir "${TARGETS}" \
    --d-min ${D_MIN} --source-name "${TRAIN_CANON}" "${DEVICE_OPTS[@]}"
  if [[ "${SKIP_CAUSAL:-false}" != true && "${SKIP_TRAIN_CAUSAL:-false}" != true ]]; then
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

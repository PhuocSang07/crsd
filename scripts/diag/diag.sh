#!/usr/bin/env bash
# Student diagnostics D1/D2/D4/D5/D6 on the track's held-out traces for one checkpoint, rows matched to the teacher
# by node hashes; D3 runs when results-proposal/<TAG>/dev-rollouts.jsonl exists.
# Usage: scripts/diag/diag.sh TRACK CKPT TAG        (CKPT = adapter dir, merged CSRD-QK dir, or "base")
set -euo pipefail
CKPT="${2:?checkpoint dir (or 'base')}"
TAG="${3:?tag}"
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
OUT="results/diag-${TAG}"
TEACHER_HO="${TEACHER_HELDOUT_WORK}/targets-dmin${D_MIN}-${SCORE}"
TEACHER_HO_CAUSAL="${TEACHER_HELDOUT_WORK}/causal-dmin${D_MIN}-${SCORE}"

MODEL="${STUDENT}"; ADAPTER_OPTS=""; RESTORE_OPTS=""
if [[ "${CKPT}" == base ]]; then
  :
elif [[ -f "${CKPT}/adapter_config.json" ]]; then
  ADAPTER_OPTS="--adapter ${CKPT}"; RESTORE_OPTS="--model-name ${STUDENT} --adapter ${CKPT} --qk-restore"
elif [[ -f "${CKPT}/config.json" ]]; then
  MODEL="${CKPT}"
  [[ -f "${CKPT}/adapters-separate/adapter_config.json" ]] && \
    RESTORE_OPTS="--model-name ${STUDENT} --adapter ${CKPT}/adapters-separate --qk-restore"
else
  echo "no adapter_config.json or config.json in ${CKPT}" >&2; exit 2
fi

HEADS="${OUT}/routing/heads-${SCORE}.json"
N_CAL=$(( $(wc -l < "${STUDENT_TRAIN_RECORDS}") < 200 ? $(wc -l < "${STUDENT_TRAIN_RECORDS}") : 200 ))
if [[ ! -f "${HEADS}" ]]; then
  rm -f "${OUT}"/routing/calib-*.npz
  python -u src/extract_routing.py --stage calibrate --model-name "${MODEL}" ${ADAPTER_OPTS} \
    --data-path "${STUDENT_TRAIN_RECORDS}" --output-dir "${OUT}/routing" --n-traces ${N_CAL} --d-min ${D_MIN} \
    2>&1 | tee "logs/diag-extract-${TAG}.log"
  python src/extract_routing.py --stage select --output-dir "${OUT}/routing" --k-per-band 16 --expected-traces ${N_CAL} \
    2>&1 | tee -a "logs/diag-extract-${TAG}.log"
fi
python -u src/extract_routing.py --stage targets --model-name "${MODEL}" ${ADAPTER_OPTS} \
  --data-path "${STUDENT_HELDOUT_RECORDS}" --heads-json "${HEADS}" --output-dir "${OUT}/targets" --d-min ${D_MIN} \
  2>&1 | tee -a "logs/diag-extract-${TAG}.log"

D5_OPTS="--distance-after ${OUT}/routing/mean-distance.npy"
BASE_DIAG="results/diag-base-${TRACK}"
[[ -f "${BASE_DIAG}/routing/mean-distance.npy" && "${CKPT}" != base ]] && D5_OPTS+=" --distance-before ${BASE_DIAG}/routing/mean-distance.npy"
[[ -n "${ADAPTER_OPTS}" ]] && D5_OPTS+=" --adapter ${CKPT}"
[[ -z "${ADAPTER_OPTS}" && -f "${CKPT}/adapters-separate/adapter_config.json" ]] && D5_OPTS+=" --adapter ${CKPT}/adapters-separate"
CAUSAL_OPTS=""; [[ -d "${TEACHER_HO_CAUSAL}" ]] && CAUSAL_OPTS="--causal-dir ${TEACHER_HO_CAUSAL}"
python src/diagnostics.py --teacher-targets "${TEACHER_HO}" --student-targets "${OUT}/targets" \
  --records "${STUDENT_HELDOUT_RECORDS}" ${CAUSAL_OPTS} ${D5_OPTS} --output-dir "${OUT}" --d-min ${D_MIN} \
  2>&1 | tee "logs/diag-${TAG}.log"

if [[ -n "${RESTORE_OPTS}" ]]; then
  python -u src/extract_routing.py --stage targets ${RESTORE_OPTS} --data-path "${STUDENT_HELDOUT_RECORDS}" \
    --heads-json "${HEADS}" --output-dir "${OUT}-qkrestore/targets" --d-min ${D_MIN} 2>&1 | tee "logs/diag-extract-${TAG}-qkrestore.log"
  python src/diagnostics.py --teacher-targets "${TEACHER_HO}" --student-targets "${OUT}-qkrestore/targets" \
    --records "${STUDENT_HELDOUT_RECORDS}" --output-dir "${OUT}-qkrestore" --d-min ${D_MIN} 2>&1 | tee "logs/diag-${TAG}-qkrestore.log"
fi

# D3: teacher-forcing on the student's own dev rollouts
ROLLOUTS="results-proposal/${TAG}/dev-rollouts.jsonl"
if [[ -f "${ROLLOUTS}" ]]; then
  D3="${OUT}-d3"
  python src/build_canonical.py --source jsonl --input "${ROLLOUTS}" --output-path "${D3}/canonical.jsonl"
  python src/data_prep.py --canonical "${D3}/canonical.jsonl" --tokenizer "${TEACHER}" --style thinking --allow-unclosed \
    --segment-mode "${SEGMENT_MODE}" --output-path "${D3}/teacher.jsonl"
  python src/data_prep.py --canonical "${D3}/canonical.jsonl" --tokenizer "${STUDENT}" --style sgl --allow-unclosed \
    --segment-mode "${SEGMENT_MODE}" --output-path "${D3}/student.jsonl"
  DEVICE_OPTS=(); [[ "${TEACHER_GPUS}" -gt 1 ]] && DEVICE_OPTS=(--device-map auto)
  sharded "d3-teacher-${TAG}" python -u src/extract_routing.py --stage targets --model-name "${TEACHER}" "${DEVICE_OPTS[@]}" \
    --data-path "${D3}/teacher.jsonl" --heads-json "${TEACHER_WORK}/routing/heads-${SCORE}.json" --output-dir "${D3}/teacher-targets" --d-min ${D_MIN}
  python -u src/extract_routing.py --stage targets --model-name "${MODEL}" ${ADAPTER_OPTS} --data-path "${D3}/student.jsonl" \
    --heads-json "${HEADS}" --output-dir "${D3}/student-targets" --d-min ${D_MIN} --save-nll
  python src/diagnostics.py --teacher-targets "${D3}/teacher-targets" --student-targets "${D3}/student-targets" \
    --records "${D3}/student.jsonl" --rollout-labels "${ROLLOUTS}.labels.jsonl" --output-dir "${D3}" --d-min ${D_MIN} \
    2>&1 | tee "logs/diag-${TAG}-d3.log"
fi
echo ">>> gates for ${TAG}: ${OUT}/diagnostics.json (G1, G2, G4), ${OUT}-d3/diagnostics.json (G3), ${OUT}-qkrestore (D6)"

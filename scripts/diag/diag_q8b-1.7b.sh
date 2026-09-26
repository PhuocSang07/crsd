#!/usr/bin/env bash
# Student diagnostics D1/D2/D4/D5/D6 on the held-out teacher traces (teacher-forced), for one checkpoint.
# Student heads are chosen on the student itself with the same receiver score, calibrated on the same D_cal as
# the teacher (first 200 train traces) -- not on the held-out traces being measured.
# Usage: scripts/diag/diag_q8b-1.7b.sh CKPT TAG     (CKPT = adapter dir, merged CSRD-QK dir, or "base")
# D6 (QK-Restore) re-reads the held-out traces with the q/k LoRA update zeroed, on the *same* student heads, so
# the RG change isolates W_Q/W_K; its pass@1 comes from eval of qk_restore.py's adapter (project_commands_pilot.sh).
# D3 (error prediction) runs when results/<TAG>/dev-rollouts.jsonl exists (eval_q8b-1.7b.sh with DEV_ROLLOUTS=1).
set -euo pipefail

CKPT="${1:?checkpoint dir (or 'base')}"
TAG="${2:?tag}"
read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
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
BASE_STUDENT="${LOCAL_MODELS_ROOT}/Qwen3-1.7B-Base"
TEACHER="${LOCAL_MODELS_ROOT}/Qwen3-8B"
D_MIN="${D_MIN:-4}"
SCORE="${SCORE:-excess_bg}"
TEACHER_HELDOUT="data/q8b/heldout-targets-dmin${D_MIN}-${SCORE}"
OUT="results/diag-${TAG}"

# What to load: base model; base + LoRA adapter; or a merged full checkpoint (CSRD-QK), whose QK-Restore
# is the main adapter kept in adapters-separate/ with its q/k update zeroed (the Q/K adapter dropped).
MODEL="${BASE_STUDENT}"; ADAPTER_OPTS=""; RESTORE_OPTS=""
if [[ "${CKPT}" == base ]]; then
  :
elif [[ -f "${CKPT}/adapter_config.json" ]]; then
  ADAPTER_OPTS="--adapter ${CKPT}"; RESTORE_OPTS="--model-name ${BASE_STUDENT} --adapter ${CKPT} --qk-restore"
elif [[ -f "${CKPT}/config.json" ]]; then
  MODEL="${CKPT}"
  [[ -f "${CKPT}/adapters-separate/adapter_config.json" ]] && \
    RESTORE_OPTS="--model-name ${BASE_STUDENT} --adapter ${CKPT}/adapters-separate --qk-restore"
else
  echo "no adapter_config.json or config.json in ${CKPT}" >&2; exit 2
fi

python -u src/extract_routing.py --stage calibrate --model-name "${MODEL}" ${ADAPTER_OPTS} \
  --data-path data/q8b/s1k-student.jsonl --output-dir "${OUT}/routing" --n-traces 200 --d-min ${D_MIN} \
  2>&1 | tee "logs/diag-extract-${TAG}.log"
python src/extract_routing.py --stage select --output-dir "${OUT}/routing" --k-per-band 16 2>&1 | tee -a "logs/diag-extract-${TAG}.log"
HEADS="${OUT}/routing/heads-${SCORE}.json"
python -u src/extract_routing.py --stage targets --model-name "${MODEL}" ${ADAPTER_OPTS} \
  --data-path data/q8b/heldout-student.jsonl --heads-json "${HEADS}" --output-dir "${OUT}/targets" --d-min ${D_MIN} \
  2>&1 | tee -a "logs/diag-extract-${TAG}.log"

D5_OPTS="--distance-after ${OUT}/routing/mean-distance.npy"
[[ -f results/diag-base/routing/mean-distance.npy && "${CKPT}" != base ]] && \
  D5_OPTS+=" --distance-before results/diag-base/routing/mean-distance.npy"
[[ -n "${ADAPTER_OPTS}" ]] && D5_OPTS+=" --adapter ${CKPT}"
[[ -z "${ADAPTER_OPTS}" && -f "${CKPT}/adapters-separate/adapter_config.json" ]] && D5_OPTS+=" --adapter ${CKPT}/adapters-separate"
python src/diagnostics.py --teacher-targets "${TEACHER_HELDOUT}" --student-targets "${OUT}/targets" \
  --records data/q8b/heldout-student.jsonl --causal-dir data/q8b/heldout-causal ${D5_OPTS} \
  --output-dir "${OUT}" --d-min ${D_MIN} 2>&1 | tee "logs/diag-${TAG}.log"

if [[ -n "${RESTORE_OPTS}" ]]; then
  # D6: same heads, q/k update zeroed
  python -u src/extract_routing.py --stage targets ${RESTORE_OPTS} --data-path data/q8b/heldout-student.jsonl \
    --heads-json "${HEADS}" --output-dir "${OUT}-qkrestore/targets" --d-min ${D_MIN} \
    2>&1 | tee "logs/diag-extract-${TAG}-qkrestore.log"
  python src/diagnostics.py --teacher-targets "${TEACHER_HELDOUT}" --student-targets "${OUT}-qkrestore/targets" \
    --records data/q8b/heldout-student.jsonl --output-dir "${OUT}-qkrestore" --d-min ${D_MIN} \
    2>&1 | tee "logs/diag-${TAG}-qkrestore.log"
fi

# D3: teacher-forcing on the student's own dev rollouts (truncated rollouts kept, without an answer node)
ROLLOUTS="results/${TAG}/dev-rollouts.jsonl"
if [[ -f "${ROLLOUTS}" ]]; then
  python src/data_prep.py --traces-path "${ROLLOUTS}" --tokenizer "${TEACHER}" --allow-unclosed --output-path "${OUT}-d3/teacher.jsonl"
  python src/data_prep.py --traces-path "${ROLLOUTS}" --tokenizer "${BASE_STUDENT}" --allow-unclosed --output-path "${OUT}-d3/student.jsonl"
  python -u src/extract_routing.py --stage targets --model-name "${TEACHER}" --data-path "${OUT}-d3/teacher.jsonl" \
    --heads-json "data/q8b/routing-teacher/heads-${SCORE}.json" --output-dir "${OUT}-d3/teacher-targets" --d-min ${D_MIN}
  python -u src/extract_routing.py --stage targets --model-name "${MODEL}" ${ADAPTER_OPTS} --data-path "${OUT}-d3/student.jsonl" \
    --heads-json "${HEADS}" --output-dir "${OUT}-d3/student-targets" --d-min ${D_MIN} --save-nll
  python src/diagnostics.py --teacher-targets "${OUT}-d3/teacher-targets" --student-targets "${OUT}-d3/student-targets" \
    --records "${OUT}-d3/student.jsonl" --rollout-labels "${ROLLOUTS}.labels.jsonl" --output-dir "${OUT}-d3" \
    --d-min ${D_MIN} 2>&1 | tee "logs/diag-${TAG}-d3.log"
fi
echo ">>> gates for ${TAG}: ${OUT}/diagnostics.json (G1, G2, G4) and ${OUT}-d3/diagnostics.json (G3)"

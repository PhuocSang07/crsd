#!/usr/bin/env bash
# Held-out mass calibration (MC-CSRD v4 Sec. 7.4, H1 / G2) -- R1-Distill-Qwen-1.5B track: every checkpoint is read on the
# 300 held-out traces with the SAME fixed student heads (b0's train heads), against the Qwen3-8B teacher (raw mass M).
#   bash scripts/diag/diag_mass_r1-qwen-1.5b.sh                  # base + every finished checkpoints/{mc,csrd}-*-r1-qwen-1.5b
#   bash scripts/diag/diag_mass_r1-qwen-1.5b.sh TAG=CKPT ...     # chosen ones (CKPT = adapter dir or "base"), e.g. epochs:
#        b0-ep1=checkpoints/mc-b0-sft-r1-qwen-1.5b/checkpoint-32
# Env: STUDENT_HEADS (b0's csrd-student-heads.json), REFERENCE (contrast arm; default mc-b1-route-l0.2 when present), GPUS.
# Output: results/diag-mass/mass-diagnostics.{md,json}; per-checkpoint targets under results/diag-mass/<tag>/ are reused.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
export TOKENIZERS_PARALLELISM=false
# Offline server: models/data come from the download.txt mirrors; no HF Hub access.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: env not found at ${PROJECT_ENV}; build it from crsd.txt (repo root) or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p logs

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
TEACHER_NAME="${LOCAL_MODELS_ROOT}/Qwen3-8B"
STUDENT_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
TEACHER_RECORDS="data/records/openr1-heldout-Qwen3-8B-thinking.jsonl"
STUDENT_RECORDS="data/records/openr1-heldout-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl"
TEACHER_HEADS="data/teacher/q8b-s1k11/routing/heads-excess_bg.json"
TEACHER_TARGETS="data/teacher/q8b-openr1-heldout/targets-dmin4-excess_bg-mc"
STUDENT_HEADS="${STUDENT_HEADS:-checkpoints/mc-b0-sft-r1-qwen-1.5b/csrd-student-heads.json}"
OUT="results/diag-mass"
D_MIN=4
BOOTSTRAP=2000

[[ -s "${STUDENT_RECORDS}" ]] || bash scripts/data/heldout_r1-qwen-1.5b.sh
[[ -f "${TEACHER_HEADS}" ]] || { echo "missing ${TEACHER_HEADS}: run scripts/teacher/teacher_qwen3-8b.sh" >&2; exit 1; }
[[ -f "${STUDENT_HEADS}" ]] || { echo "missing ${STUDENT_HEADS}: train ARM=b0 (scripts/csrd/mc_csrd_lora_r1-qwen-1.5b.sh)" >&2; exit 1; }

extract() {  # extract MODEL ADAPTER_DIR|"" HEADS RECORDS OUTPUT_DIR LOG
  local n_records n_done
  n_records=$(wc -l < "$4")
  n_done=$(ls "$5"/*.npz 2>/dev/null | wc -l)
  if [[ "${n_done}" -ge "${n_records}" ]]; then echo "skip $5 (${n_done} traces)"; return; fi
  OPTS=""
  OPTS+=" --stage targets"
  OPTS+=" --model-name $1"
  [[ -n "$2" ]] && OPTS+=" --adapter $2"
  OPTS+=" --heads-json $3"
  OPTS+=" --data-path $4"
  OPTS+=" --output-dir $5"
  OPTS+=" --d-min ${D_MIN}"
  CMD="python -u ${BASE_PATH}/src/extract_routing.py ${OPTS}"
  echo "${CMD}"
  ${CMD} 2>&1 | tee "$6"
}

extract "${TEACHER_NAME}" "" "${TEACHER_HEADS}" "${TEACHER_RECORDS}" "${TEACHER_TARGETS}" logs/diag-mass-teacher.log

declare -A CKPTS=()
if [[ $# -gt 0 ]]; then
  for item in "$@"; do CKPTS["${item%%=*}"]="${item#*=}"; done
else
  CKPTS[base]=base
  for ckpt in checkpoints/mc-*-r1-qwen-1.5b checkpoints/csrd-*-r1-qwen-1.5b; do
    [[ -f "${ckpt}/adapter_config.json" ]] || continue
    CKPTS["$(basename "${ckpt}" -r1-qwen-1.5b)"]="${ckpt}"
  done
fi

STUDENT_OPTS=()
for tag in $(printf '%s\n' "${!CKPTS[@]}" | sort); do
  ckpt="${CKPTS[${tag}]}"
  adapter=""
  [[ "${ckpt}" != base ]] && adapter="${ckpt}"
  extract "${STUDENT_NAME}" "${adapter}" "${STUDENT_HEADS}" "${STUDENT_RECORDS}" "${OUT}/${tag}/targets" "logs/diag-mass-${tag}.log"
  STUDENT_OPTS+=(--student "${tag}=${OUT}/${tag}/targets")
done

REFERENCE="${REFERENCE:-mc-b1-route-l0.2}"
REF_OPTS=()
[[ -n "${CKPTS[${REFERENCE}]:-}" ]] && REF_OPTS=(--reference "${REFERENCE}")
python "${BASE_PATH}/src/mass_diagnostics.py" --teacher "${TEACHER_TARGETS}" "${STUDENT_OPTS[@]}" "${REF_OPTS[@]}" \
  --output-dir "${OUT}" --d-min ${D_MIN} --bootstrap ${BOOTSTRAP} 2>&1 | tee logs/diag-mass.log

echo ">>> STOP AND READ ${OUT}/mass-diagnostics.md: H1 holds if mc-b3 has a lower E_Z / KL_raw than ${REFERENCE} with the"
echo ">>> paired CI below 0, and not only because of the weaker pressure (mc-b4 row)."

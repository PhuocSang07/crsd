#!/usr/bin/env bash
# Ablation: the teacher WRITES its own traces on the s1K questions and reads them (teacher = author, proposal
# Sec. 6.1); the student still trains on the SGL format. Compare with project_commands_read.sh (main): same
# students, same config, but the data differ from the baselines', so baselines would have to be retrained here.
#   gen-q8b-1.7b    Qwen3-8B writes s1K-Q8B            -> Qwen3-1.7B-Base
#   gen-d32b-q8b    R1-Distill-Qwen-32B writes s1K-D32B -> Qwen3-8B
# Comment out any line you don't want; every stage skips what already exists.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
TRACKS=(${TRACKS:-gen-q8b-1.7b gen-d32b-q8b})
SEEDS=(${SEEDS:-42 43 44})
LAMBDAS=(${LAMBDAS:-0.3 1})
PILOT_N="aime24=8,aime25=8,amc12=8"   # Table 6; drop N_SAMPLES_MAP for the main table (n = 16)

for TRACK in "${TRACKS[@]}"; do
  # ===================== DATA + TEACHER SIGNALS =====================
  bash scripts/gen/gen_traces.sh "${TRACK}"          # 8 traces per s1K question + 300 held-out (G0 >= 600)
  bash scripts/data/canonical.sh "${TRACK}"
  bash scripts/data/records.sh "${TRACK}"            # teacher (thinking) / student (SGL) renderings, anchors
  bash scripts/targets/teacher_signals.sh "${TRACK}" # heads, P/Z, causal -> one reusable signal bank

  # ===================== WEEK 1: DIAGNOSTICS =====================
  bash scripts/train/train.sh "${TRACK}" sft 42
  bash scripts/diag/diag.sh "${TRACK}" base "base-${TRACK}"
  DEV_ROLLOUTS=1 N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval.sh "${TRACK}" "checkpoints/sft-${TRACK}-s42" "sft-${TRACK}-s42"
  bash scripts/diag/diag.sh "${TRACK}" "checkpoints/sft-${TRACK}-s42" "sft-${TRACK}-s42"
  "${PROJECT_ENV}/bin/python" src/qk_restore.py --adapter "checkpoints/sft-${TRACK}-s42" --output-dir "checkpoints/sft-qkrestore-${TRACK}-s42"
  N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval.sh "${TRACK}" "checkpoints/sft-qkrestore-${TRACK}-s42" "sft-qkrestore-${TRACK}-s42"
  echo ">>> STOP AND READ results/diag-sft-${TRACK}-s42/diagnostics.md (G1, G2, G4) and ...-d3 (G3)."

  # ===================== WEEK 2: INTERVENTION =====================
  for seed in "${SEEDS[@]}"; do
    bash scripts/train/train.sh "${TRACK}" sft "${seed}"
    for lam in "${LAMBDAS[@]}"; do LAMBDA="${lam}" bash scripts/train/train.sh "${TRACK}" csrd "${seed}"; done
  done
  for seed in "${SEEDS[@]}"; do
    N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval.sh "${TRACK}" "checkpoints/sft-${TRACK}-s${seed}" "sft-${TRACK}-s${seed}"
    for lam in "${LAMBDAS[@]}"; do
      tag="csrd-l${lam}-${TRACK}-s${seed}"
      N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval.sh "${TRACK}" "checkpoints/${tag}" "${tag}"
    done
  done
  PROMPT_STYLE=zeroshot N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval.sh "${TRACK}" base "base-zeroshot-${TRACK}"   # B0
  PROMPT_STYLE=fewshot N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval.sh "${TRACK}" base "base-fewshot-${TRACK}"
  tag="csrd-l0.3-${TRACK}-s42"
  bash scripts/inject/inject.sh "${TRACK}" "checkpoints/sft-${TRACK}-s42" "sft-${TRACK}-s42"
  bash scripts/inject/inject.sh "${TRACK}" "checkpoints/${tag}" "${tag}"
  bash scripts/diag/diag.sh "${TRACK}" "checkpoints/${tag}" "${tag}"
  "${PROJECT_ENV}/bin/python" src/qk_restore.py --adapter "checkpoints/${tag}" --output-dir "checkpoints/csrd-qkrestore-l0.3-${TRACK}-s42"
  N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval.sh "${TRACK}" "checkpoints/csrd-qkrestore-l0.3-${TRACK}-s42" "csrd-qkrestore-l0.3-${TRACK}-s42"
done

# =========================== COMPARE ==========================
for protocol in proposal palign; do
  for TRACK in "${TRACKS[@]}"; do
    "${PROJECT_ENV}/bin/python" src/compare_results.py --results-dir "results-${protocol}" --track "${TRACK}" || true
  done
done
for TRACK in "${TRACKS[@]}"; do "${PROJECT_ENV}/bin/python" src/pilot_report.py --track "${TRACK}"; done

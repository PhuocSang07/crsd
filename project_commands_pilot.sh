#!/usr/bin/env bash
# CSRD pilot (proposal Sec. 7, Table 6): Qwen3-8B (thinking) -> Qwen3-1.7B-Base, s1K-Q8B, LoRA r = 64.
# Week 1 = data + diagnostics (gates G0-G4), week 2 = intervention (G5, G6). Comment out any line you
# don't want to run; every stage skips work whose outputs already exist.
#
# Decision tree after week 1/2:
#   G1 & G3 & G5      -> full programme (4B student, baselines B2-B7, ablations A1-A15)
#   G1, not G4        -> switch the default to CSRD-C (scripts/targets/causal_heads_qwen3-8b.sh, arm csrd-c), redo week 2
#   G1, not G3/G5     -> fallback A (Sec. 11.2); keep the diagnostics as analysis
#   not G1            -> stop CSRD, fallback B (analysis paper)
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
SEEDS=(42 43 44)
LAMBDAS=(0.3 1)

# ======================= WEEK 1: DATA =======================
bash scripts/gen/gen_s1k-q8b.sh          # G0: >= 600 questions with a correct trace (stats.json)
bash scripts/gen/gen_heldout-q8b.sh      # 300 held-out teacher traces (MATH train L3-5, disjoint from dev/test)
bash scripts/data/data_q8b-1.7b.sh       # nodes (\n\n, >= 40 chars, <= 400) for both tokenizers, anchor labels, 13-gram check
bash scripts/targets/targets_qwen3-8b.sh # receiver heads (both scores), P/Z targets, causal targets (20% train, 20 held-out)

# ==================== WEEK 1: DIAGNOSTICS ====================
bash scripts/train/train_q8b-1.7b.sh sft 42
bash scripts/diag/diag_q8b-1.7b.sh base base
DEV_ROLLOUTS=1 N_SAMPLES_MAP=aime24=8,aime25=8,amc12=8 bash scripts/eval/eval_q8b-1.7b.sh checkpoints/sft-q8b-1.7b-s42 sft-q8b-1.7b-s42
bash scripts/diag/diag_q8b-1.7b.sh checkpoints/sft-q8b-1.7b-s42 sft-q8b-1.7b-s42
# D6 pass@1: the SFT student with its q/k LoRA update zeroed (its RG is in results/diag-sft-q8b-1.7b-s42-qkrestore)
"${PROJECT_ENV}/bin/python" src/qk_restore.py --adapter checkpoints/sft-q8b-1.7b-s42 --output-dir checkpoints/sft-qkrestore-q8b-1.7b-s42
N_SAMPLES_MAP=aime24=8,aime25=8,amc12=8 bash scripts/eval/eval_q8b-1.7b.sh checkpoints/sft-qkrestore-q8b-1.7b-s42 sft-qkrestore-q8b-1.7b-s42
echo ">>> STOP AND READ results/diag-sft-q8b-1.7b-s42/diagnostics.md: G1 (routing gap grows with distance),"
echo "    G2 (anchors), G4 (attention-causal Spearman >= 0.3) and results/diag-sft-q8b-1.7b-s42-d3 (G3)."

# ==================== WEEK 2: INTERVENTION ====================
for seed in "${SEEDS[@]}"; do
  bash scripts/train/train_q8b-1.7b.sh sft "${seed}"
  for lam in "${LAMBDAS[@]}"; do
    LAMBDA="${lam}" bash scripts/train/train_q8b-1.7b.sh csrd "${seed}"
  done
done

deactivate 2>/dev/null || true
unset VIRTUAL_ENV
PILOT_N="aime24=8,aime25=8,amc12=8"   # MATH500 keeps n = 4
for seed in "${SEEDS[@]}"; do
  N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval_q8b-1.7b.sh "checkpoints/sft-q8b-1.7b-s${seed}" "sft-q8b-1.7b-s${seed}"
  for lam in "${LAMBDAS[@]}"; do
    tag="csrd-l${lam}-q8b-1.7b-s${seed}"
    N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval_q8b-1.7b.sh "checkpoints/${tag}" "${tag}"
  done
done
PROMPT_STYLE=zeroshot N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval_q8b-1.7b.sh base base-zeroshot   # B0
PROMPT_STYLE=fewshot N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval_q8b-1.7b.sh base base-fewshot

# error injection on 100 cases + mechanism check on the best CSRD arm
bash scripts/inject/inject_q8b-1.7b.sh checkpoints/sft-q8b-1.7b-s42 sft-q8b-1.7b-s42
bash scripts/inject/inject_q8b-1.7b.sh checkpoints/csrd-l0.3-q8b-1.7b-s42 csrd-l0.3-q8b-1.7b-s42
bash scripts/diag/diag_q8b-1.7b.sh checkpoints/csrd-l0.3-q8b-1.7b-s42 csrd-l0.3-q8b-1.7b-s42
# Hypothesis 4(iii): CSRD's gain over SFT should shrink by >= half under QK-Restore (compare with sft-qkrestore)
"${PROJECT_ENV}/bin/python" src/qk_restore.py --adapter checkpoints/csrd-l0.3-q8b-1.7b-s42 --output-dir checkpoints/csrd-qkrestore-l0.3-q8b-1.7b-s42
N_SAMPLES_MAP="${PILOT_N}" bash scripts/eval/eval_q8b-1.7b.sh checkpoints/csrd-qkrestore-l0.3-q8b-1.7b-s42 csrd-qkrestore-l0.3-q8b-1.7b-s42

# =========================== COMPARE ==========================
# G5: mean pass@1 +1.5 over SFT (or error detection +5 with no pass@1 loss: results/inject-*/report.json).
# G6: throughput overhead <= 25% -- compare train_runtime in checkpoints/*/run-summary.json (csrd vs sft).
"${PROJECT_ENV}/bin/python" src/compare_results.py --results-dir results --baseline sft-q8b-1.7b
"${PROJECT_ENV}/bin/python" src/pilot_report.py --baseline sft-q8b-1.7b --arm csrd-l0.3-q8b-1.7b

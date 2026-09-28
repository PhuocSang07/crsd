#!/usr/bin/env bash
# Eval under the 32k generation cap (SGL results-32k convention) on MATH500 + AIME24 -- DeepSeek-R1-Distill-Qwen-1.5B track.
#   bash scripts/eval/eval32k_r1-qwen-1.5b.sh               # every finished checkpoints/*-r1-qwen-1.5b
#   bash scripts/eval/eval32k_r1-qwen-1.5b.sh CKPT [TAG]    # one checkpoint (adapter dir, full model dir, or "base")
# Same sampling as the 4k runs (T 0.6, top-p 0.9, rep 1.05, n 3, seed 42, eager); only the cap changes: max_model_len
# 32768, max_tokens 30720. Results go to results-32k/<tag>/ (results-palign/ is kept); a tag with a summary.json is
# skipped. SFT reference: copy SGL's checkpoints/vanilla-r1-qwen-1.5b into checkpoints/ first, or train mc-b0-sft.
# Env: BENCHMARKS (math500,aime24; the MC-CSRD driver uses aime24,aime25 into RESULTS_DIR=results-32k-aime), N_SAMPLES (3),
# EVAL_BASE=1 also evaluates the untrained student (tag base-r1-qwen-1.5b, the zero-shot row of v4 Stage A),
# BASELINE (vanilla-r1-qwen-1.5b) is the arm compare_results.py contrasts with, FAMILY its Holm family (regex),
# CONTRASTS extra paired contrasts "armA:armB,..." (full tags).
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
export GPUS="${GPUS:-0}"
export RESULTS_DIR="${RESULTS_DIR:-${BASE_PATH}/results-32k}"
export BENCHMARKS="${BENCHMARKS:-math500,aime24}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-30720}"
mkdir -p "${RESULTS_DIR}" logs

run_eval() {  # run_eval <model path> <tag>
  if [[ -f "${RESULTS_DIR}/$2/summary.json" ]]; then
    echo "[skip] $2: ${RESULTS_DIR}/$2/summary.json exists"
    return
  fi
  echo "[$(date +%H:%M)] [eval 32k] $2 <- $1"
  EVAL_LOG="${BASE_PATH}/logs/eval32k-$2.log" bash scripts/eval/eval_r1-qwen-1.5b.sh "$1" "$2"
}

if [[ -n "${1:-}" ]]; then
  run_eval "$1" "${2:-$(basename "$1")}"
else
  [[ "${EVAL_BASE:-0}" == 1 ]] && run_eval base base-r1-qwen-1.5b
  for ckpt in checkpoints/*-r1-qwen-1.5b; do
    # a finished run has the adapter or a merged model at its root; checkpoint-N/ dirs are not evaluated
    [[ -f "${ckpt}/adapter_config.json" || -f "${ckpt}/config.json" ]] || continue
    run_eval "${ckpt}" "$(basename "${ckpt}")"
  done
fi

CONTRAST_OPTS=()
[[ -n "${CONTRASTS:-}" ]] && CONTRAST_OPTS=(--contrasts "${CONTRASTS}")
PYTHONPATH="${BASE_PATH}/src" "${PROJECT_ENV:-/mnt/local/uvenvs/crsd}/bin/python" src/compare_results.py \
  --results-dir "${RESULTS_DIR}" --track r1-qwen-1.5b --baseline "${BASELINE:-vanilla-r1-qwen-1.5b}" \
  --family "${FAMILY:-^csrd-(?!qkrestore)}" "${CONTRAST_OPTS[@]}" || true

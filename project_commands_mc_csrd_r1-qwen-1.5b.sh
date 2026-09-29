#!/usr/bin/env bash
# MC-CSRD v4 driver (MC_CSRD_proposal_v4.tex Sec. 6-8) -- DeepSeek-R1-Distill-Qwen-1.5B student, Qwen3-8B reader teacher,
# s1K-1.1, one GPU. G0 unit tests -> raw-mass teacher bank -> per arm (b0 first): train, P-ALIGN 4k eval on all four
# benchmarks (results-palign/), 32k eval on AIME24 + AIME25 (results-32k-aime/) -> the untrained student under both
# protocols -> paired comparisons (H2) -> held-out mass diagnostics on fixed heads (H1). Finished steps are skipped.
# Usage: GPUS="0" bash project_commands_mc_csrd_r1-qwen-1.5b.sh
# New server: scripts/setup.sh (needs PyPI) -> scripts/data/download_r1-qwen-1.5b.sh (needs HF Hub) -> this (offline);
# the held-out diagnostics read data/canonical/openr1-heldout.jsonl (tracked in the repo).
# Env: ARMS (order of b0 b1 b2 b3 b4; b0 first, its heads and gradient probe feed the others), MC_LAMBDA (0.2, band
# mean), EVAL_BASE (1: also evaluate the untrained student), N_SAMPLES_32K (3), GPU_MEM_UTIL (vLLM, 0.9),
# LOCAL_MODELS_ROOT (/path/models), LOCAL_DATA_ROOT / BENCH_DATA_ROOT (auto-detected, see PATHS), PROJECT_ENV.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# ============================ PATHS ============================
# Models: /path/models (the H200 layout the 28/09 runs used). Data and the four test sets: LOCAL_DATA_ROOT /
# BENCH_DATA_ROOT when set, else the first candidate holding aime24/. Exported, so every sub-script reads the same paths.
export PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
export LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/path/models}"
if [[ -z "${LOCAL_DATA_ROOT:-}" ]]; then
  for candidate in /path/data /path/datasets /path/models /path /mnt/local/_data/aiskylimit_new_nothingnew_2; do
    [[ -d "${candidate}/aime24" ]] && { LOCAL_DATA_ROOT="${candidate}"; break; }
  done
fi
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-${LOCAL_DATA_ROOT}}"
missing=()
[[ -x "${PROJECT_ENV}/bin/python" ]] || missing+=("${PROJECT_ENV}/bin/python (PROJECT_ENV)")
for model in DeepSeek-R1-Distill-Qwen-1.5B Qwen3-8B; do
  [[ -d "${LOCAL_MODELS_ROOT}/${model}" ]] || missing+=("${LOCAL_MODELS_ROOT}/${model} (LOCAL_MODELS_ROOT)")
done
if [[ -n "${BENCH_DATA_ROOT}" ]]; then  # "" = read the test sets from the HF cache instead
  for bench in aime24 aime25 aimo-validation-amc MATH-500; do
    [[ -d "${BENCH_DATA_ROOT}/${bench}" ]] || missing+=("${BENCH_DATA_ROOT}/${bench} (BENCH_DATA_ROOT / LOCAL_DATA_ROOT)")
  done
fi
if (( ${#missing[@]} )); then
  printf 'missing: %s\n' "${missing[@]}" >&2
  echo "set PROJECT_ENV / LOCAL_MODELS_ROOT / LOCAL_DATA_ROOT (or BENCH_DATA_ROOT) to where they are on this server" >&2
  exit 1
fi
echo "models ${LOCAL_MODELS_ROOT} | data ${LOCAL_DATA_ROOT} | benchmarks ${BENCH_DATA_ROOT:-HF cache} | env ${PROJECT_ENV}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"
export CSRD_GRAD_LOG_INTERVAL="${CSRD_GRAD_LOG_INTERVAL:-10}"
MC_LAMBDA="${MC_LAMBDA:-0.2}"
ARMS="${ARMS:-b0 b3 b1 b4 b2}"
EVAL_BASE="${EVAL_BASE:-1}"
N_SAMPLES_32K="${N_SAMPLES_32K:-3}"
PY="${PROJECT_ENV}/bin/python"
TRACK=r1-qwen-1.5b
RESULTS_4K=results-palign
RESULTS_32K=results-32k-aime
B0_TAG="mc-b0-sft-${TRACK}"

b4_lambda() {  # MC_LAMBDA x median ||grad L_mc_raw|| / ||grad (L_route + 0.1 L_mass)|| at the first post-warmup step (b0)
  "${PY}" - "checkpoints/${B0_TAG}/csrd-norm-probe.json" "${MC_LAMBDA}" <<'PY'
import json, sys
probe = json.load(open(sys.argv[1]))
print(f"{float(sys.argv[2]) * probe['median_ratio']['mc_raw/route_mass0.1']:.3g}")
PY
}

tag_of() {  # tag_of ARM LAMBDA -- same rule as scripts/csrd/mc_csrd_lora_r1-qwen-1.5b.sh
  case "$1" in
    b0) echo "mc-b0-sft-${TRACK}" ;;
    b1) echo "mc-b1-route-l$2-${TRACK}" ;;
    b2) echo "mc-b2-syn-l$2-${TRACK}" ;;
    b3) echo "mc-b3-raw-l$2-${TRACK}" ;;
    b4) echo "mc-b4-route-l$2-${TRACK}" ;;
  esac
}

eval_4k() {  # eval_4k CKPT TAG -- P-ALIGN protocol (T 0.6, top-p 0.9, rep 1.05, n 3), AIME24/25, AMC12, MATH500
  [[ -f "${RESULTS_4K}/$2/summary.json" ]] && { echo "[skip] 4k $2"; return; }
  GPUS="${GPUS%% *}" RESULTS_DIR="${RESULTS_4K}" bash scripts/eval/eval_r1-qwen-1.5b.sh "$1" "$2"
}

eval_32k() {  # eval_32k CKPT TAG -- same sampling, cap 30720 generated tokens, AIME24 + AIME25 only
  GPUS="${GPUS%% *}" RESULTS_DIR="${RESULTS_32K}" BENCHMARKS=aime24,aime25 N_SAMPLES="${N_SAMPLES_32K}" \
    BASELINE="${B0_TAG}" FAMILY='^mc-b[1-4]' bash scripts/eval/eval32k_r1-qwen-1.5b.sh "$1" "$2"
}

# ============================ G0: implementation checks ============================
# Identities of v4 Sec. 4-5 (chain rule, flat = hierarchical loss and gradient, raw vs synthetic pooling, unbiased query
# weights), q/k capture vs eager attention (Qwen2 student, Qwen3 teacher), trainer objectives and probe.
mkdir -p logs
PYTHONPATH="${BASE}/src" "${PY}" -m pytest tests -q -p no:cacheprovider 2>&1 | tee logs/mc-g0-tests.log | tail -3 || {
  echo "G0 failed (logs/mc-g0-tests.log): fix the implementation before any training" >&2; exit 1; }

# ============================ DATA ============================
[[ -s data/records/s1k11-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl ]] || bash scripts/data/data_r1-qwen-1.5b.sh
# Teacher bank with the raw mass M (v3 banks lack it); the teacher heads of the v3 bank are reused.
[[ -s signals/q8b-s1k11-dmin4-excess_bg-mc.safetensors ]] || BANK_SUFFIX=-mc bash scripts/teacher/teacher_qwen3-8b.sh

# ======================= TRAIN + EVAL 4k + EVAL 32k (per arm) ======================
for arm in ${ARMS}; do
  lambda=${MC_LAMBDA}
  [[ "${arm}" == b4 ]] && lambda="$(b4_lambda)"
  TAG="$(tag_of "${arm}" "${lambda}")"
  echo "[$(date +%H:%M)] ===== ${arm}: ${TAG}"
  ARM=${arm} MC_LAMBDA=${lambda} bash scripts/csrd/mc_csrd_lora_r1-qwen-1.5b.sh
  eval_4k "checkpoints/${TAG}" "${TAG}"
  eval_32k "checkpoints/${TAG}" "${TAG}"
done
# the untrained student under both protocols (v4 Stage A, zero-shot row)
if [[ "${EVAL_BASE}" == 1 ]]; then
  eval_4k base "base-${TRACK}"
  eval_32k base "base-${TRACK}"
fi

# =========================== COMPARE (H2) ==========================
# vs b0 (SFT) for every arm; b3 also vs b1 / b2 / b4. Per benchmark: paired delta pass@1, bootstrap CI over problems.
B1_TAG="$(tag_of b1 "${MC_LAMBDA}")"
B3_TAG="$(tag_of b3 "${MC_LAMBDA}")"
B4_TAG=""; [[ -f "checkpoints/${B0_TAG}/csrd-norm-probe.json" ]] && B4_TAG="$(tag_of b4 "$(b4_lambda)")"
CONTRASTS="${B3_TAG}:${B1_TAG},${B3_TAG}:$(tag_of b2 "${MC_LAMBDA}")${B4_TAG:+,${B3_TAG}:${B4_TAG}}"
for results in "${RESULTS_4K}" "${RESULTS_32K}"; do
  PYTHONPATH="${BASE}/src" "${PY}" src/compare_results.py --results-dir "${results}" --track "${TRACK}" \
    --baseline "${B0_TAG}" --family '^mc-b[1-4]' --contrasts "${CONTRASTS}" || true
done

# ======================= HELD-OUT MASS DIAGNOSTICS (H1) ======================
if [[ -s data/canonical/openr1-heldout.jsonl || -s data/records/openr1-heldout-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl ]]; then
  GPUS="${GPUS%% *}" REFERENCE="${B1_TAG%-${TRACK}}" bash scripts/diag/diag_mass_r1-qwen-1.5b.sh
else
  echo "skip the held-out diagnostics: data/canonical/openr1-heldout.jsonl is missing (git checkout it)"
fi
echo ">>> results: ${RESULTS_4K}/comparison-table-${TRACK}.md (4k), ${RESULTS_32K}/comparison-table-${TRACK}.md (32k AIME),"
echo ">>> contrasts-${TRACK}.json in both, results/diag-mass/mass-diagnostics.md (H1)."

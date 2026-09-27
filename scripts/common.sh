#!/usr/bin/env bash
# Shared track table and environment for every phase script: `source scripts/common.sh TRACK`.
# Teacher signals depend only on (teacher, data): one signal bank is reused by any student.
set -euo pipefail

TRACK="${1:?track is required: read-q8b-1.7b | read-q8b-r1.5b | read-d32b-q8b | gen-q8b-1.7b | gen-d32b-q8b}"
case "${TRACK}" in
  read-q8b-1.7b) TT=q8b;  TEACHER_DIR=Qwen3-8B;                     STUDENT_DIR=Qwen3-1.7B-Base; MODE=read ;;
  read-q8b-r1.5b) TT=q8b; TEACHER_DIR=Qwen3-8B;                     STUDENT_DIR=DeepSeek-R1-Distill-Qwen-1.5B; MODE=read ;;
  read-d32b-q8b) TT=d32b; TEACHER_DIR=DeepSeek-R1-Distill-Qwen-32B; STUDENT_DIR=Qwen3-8B;        MODE=read ;;
  gen-q8b-1.7b)  TT=q8b;  TEACHER_DIR=Qwen3-8B;                     STUDENT_DIR=Qwen3-1.7B-Base; MODE=gen ;;
  gen-d32b-q8b)  TT=d32b; TEACHER_DIR=DeepSeek-R1-Distill-Qwen-32B; STUDENT_DIR=Qwen3-8B;        MODE=gen ;;
  *) echo "unknown track: ${TRACK}" >&2; exit 2 ;;
esac

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # never build the venv here: scripts/setup.sh needs PyPI (the offline server builds it from crsd.txt)
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: env not found at ${PROJECT_ENV}; build it from crsd.txt (repo root) or scripts/setup.sh, or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# Offline server: models/data come from the download.txt mirrors; no HF Hub access, no vLLM usage stats.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-${HF_HUB_OFFLINE}}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
# Local s1K/test-set mirrors for benchmarks.py and decontamination; BENCH_DATA_ROOT="" uses the HF Hub.
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-${LOCAL_DATA_ROOT}}"
read -ra GPUS <<< "${GPUS:-0}"
mkdir -p logs data/canonical data/records signals

TEACHER="${LOCAL_MODELS_ROOT}/${TEACHER_DIR}"
STUDENT="${LOCAL_MODELS_ROOT}/${STUDENT_DIR}"
JUDGE="${LOCAL_MODELS_ROOT}/Qwen3-8B"            # non-thinking LLM judge for free-form answers
# GPUs per teacher process: the 32B teacher is sharded over 2.
TEACHER_GPUS=1; [[ "${TT}" == d32b ]] && TEACHER_GPUS=2
TEACHER_GPUS="${TEACHER_GPUS_OVERRIDE:-${TEACHER_GPUS}}"

if [[ "${MODE}" == read ]]; then
  TRAIN_CANON=s1k11                 # simplescaling/s1K-1.1 (DeepSeek-R1 traces)
  HELDOUT_CANON=openr1-heldout      # 300 OpenR1-Math R1 traces, 13-gram-disjoint from s1K/tests
else
  TRAIN_CANON="gen-${TT}-s1k"       # the teacher's own traces on the s1K questions
  HELDOUT_CANON="gen-${TT}-heldout" # its own traces on MATH-train L3-5 (disjoint from dev/s1K/tests)
fi
SEGMENT_MODE="${SEGMENT_MODE:-paragraph}"
D_MIN="${D_MIN:-4}"
SCORE="${SCORE:-excess_bg}"
SEG_TAG=""; [[ "${SEGMENT_MODE}" != paragraph ]] && SEG_TAG="-${SEGMENT_MODE}"

records_path() {  # records_path CANON MODEL_DIR STYLE
  echo "data/records/$1${SEG_TAG}-$2-$3.jsonl"
}
TEACHER_TRAIN_RECORDS="$(records_path "${TRAIN_CANON}" "${TEACHER_DIR}" thinking)"
TEACHER_HELDOUT_RECORDS="$(records_path "${HELDOUT_CANON}" "${TEACHER_DIR}" thinking)"
STUDENT_TRAIN_RECORDS="$(records_path "${TRAIN_CANON}" "${STUDENT_DIR}" sgl)"
STUDENT_HELDOUT_RECORDS="$(records_path "${HELDOUT_CANON}" "${STUDENT_DIR}" sgl)"
TEACHER_WORK="data/teacher/${TT}-${TRAIN_CANON}${SEG_TAG}"
TEACHER_HELDOUT_WORK="data/teacher/${TT}-${HELDOUT_CANON}${SEG_TAG}"
SIGNALS="signals/${TT}-${TRAIN_CANON}${SEG_TAG}-dmin${D_MIN}-${SCORE}.safetensors"

# One teacher process per group of TEACHER_GPUS GPUs.
GPU_GROUPS=()
for (( g = 0; g + TEACHER_GPUS <= ${#GPUS[@]}; g += TEACHER_GPUS )); do
  GPU_GROUPS+=("$(IFS=,; echo "${GPUS[*]:g:TEACHER_GPUS}")")
done
[[ ${#GPU_GROUPS[@]} -gt 0 ]] || { echo "need at least ${TEACHER_GPUS} GPU(s) in GPUS" >&2; exit 2; }

sharded() {  # sharded NAME cmd...: one shard per GPU group, wait for all
  local name=$1; shift
  local pids=() fail=0 n=${#GPU_GROUPS[@]}
  for i in "${!GPU_GROUPS[@]}"; do
    echo "[${GPU_GROUPS[$i]}] $* --num-shards ${n} --shard-index ${i}"
    CUDA_VISIBLE_DEVICES="${GPU_GROUPS[$i]}" "$@" --num-shards "${n}" --shard-index "${i}" > "logs/${name}-shard${i}.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "shard ${i} failed: logs/${name}-shard${i}.log" >&2; fail=1; }; done
  return ${fail}
}

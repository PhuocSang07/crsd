#!/usr/bin/env bash
# Shared track table and environment for every phase script: `source scripts/common.sh TRACK`.
#
# Tracks (teacher -> student, training data):
#   read-q8b-1.7b   Qwen3-8B reads s1K-1.1 (R1 traces)            -> Qwen3-1.7B-Base   (main: same data as the baselines)
#   read-d32b-q8b   R1-Distill-Qwen-32B reads s1K-1.1 (R1 traces)  -> Qwen3-8B          (main: same student as P-ALIGN/SSFT)
#   gen-q8b-1.7b    Qwen3-8B writes s1K-Q8B and reads it          -> Qwen3-1.7B-Base   (ablation: teacher = author)
#   gen-d32b-q8b    R1-Distill-Qwen-32B writes s1K-D32B, reads it  -> Qwen3-8B          (ablation: teacher = author)
#
# Layout. Content and teacher signals depend only on (teacher, data), never on the student, so a signal
# bank computed once (e.g. by the 32B teacher) is reused by any later student:
#   data/canonical/<CANON>.jsonl                        question/thinking/answer (build_canonical.py)
#   data/records/<CANON>-<model>-<style>.jsonl          one model's rendering + node token spans
#   data/teacher/<TT>-<CANON>/{routing,targets,causal}  teacher extraction (resumable per-trace .npz)
#   signals/<TT>-<CANON>-dmin<D>-<SCORE>.safetensors     packed, reusable signal bank
#   checkpoints/<arm>-<track>-s<seed>, results-{proposal,palign}/<tag>
set -euo pipefail

TRACK="${1:?track is required: read-q8b-1.7b | read-d32b-q8b | gen-q8b-1.7b | gen-d32b-q8b}"
case "${TRACK}" in
  read-q8b-1.7b) TT=q8b;  TEACHER_DIR=Qwen3-8B;                     STUDENT_DIR=Qwen3-1.7B-Base; MODE=read ;;
  read-d32b-q8b) TT=d32b; TEACHER_DIR=DeepSeek-R1-Distill-Qwen-32B; STUDENT_DIR=Qwen3-8B;        MODE=read ;;
  gen-q8b-1.7b)  TT=q8b;  TEACHER_DIR=Qwen3-8B;                     STUDENT_DIR=Qwen3-1.7B-Base; MODE=gen ;;
  gen-d32b-q8b)  TT=d32b; TEACHER_DIR=DeepSeek-R1-Distill-Qwen-32B; STUDENT_DIR=Qwen3-8B;        MODE=gen ;;
  *) echo "unknown track: ${TRACK}" >&2; exit 2 ;;
esac

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || bash "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
# benchmarks.py / the 13-gram filter read s1K and the test sets from here; BENCH_DATA_ROOT="" = HF Hub
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT-${LOCAL_DATA_ROOT}}"
read -ra GPUS <<< "${GPUS:-0}"
mkdir -p logs data/canonical data/records signals

TEACHER="${LOCAL_MODELS_ROOT}/${TEACHER_DIR}"
STUDENT="${LOCAL_MODELS_ROOT}/${STUDENT_DIR}"
JUDGE="${LOCAL_MODELS_ROOT}/Qwen3-8B"            # non-thinking LLM judge for free-form answers
# GPUs per teacher process: the 32B teacher (64 GB of bf16 weights) is sharded over 2 GPUs.
TEACHER_GPUS=1; [[ "${TT}" == d32b ]] && TEACHER_GPUS=2
TEACHER_GPUS="${TEACHER_GPUS_OVERRIDE:-${TEACHER_GPUS}}"

if [[ "${MODE}" == read ]]; then
  TRAIN_CANON=s1k11                 # simplescaling/s1K-1.1, DeepSeek-R1 trajectory + attempt
  HELDOUT_CANON=openr1-heldout      # 300 R1 traces from OpenR1-Math, 13-gram-disjoint from s1K and the tests
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

# Groups of TEACHER_GPUS GPUs, one teacher process per group (CUDA_VISIBLE_DEVICES strings).
GPU_GROUPS=()
for (( g = 0; g + TEACHER_GPUS <= ${#GPUS[@]}; g += TEACHER_GPUS )); do
  GPU_GROUPS+=("$(IFS=,; echo "${GPUS[*]:g:TEACHER_GPUS}")")
done
[[ ${#GPU_GROUPS[@]} -gt 0 ]] || { echo "need at least ${TEACHER_GPUS} GPU(s) in GPUS" >&2; exit 2; }

sharded() {  # sharded NAME cmd...: one process per GPU group with --num-shards/--shard-index, wait for all
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

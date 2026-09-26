#!/usr/bin/env bash
# Train one arm of the Qwen3-8B -> Qwen3-1.7B-Base track. Every arm shares data, LoRA (all-linear,
# r = alpha = 64, dropout 0), lr / cosine schedule, global batch 16, epochs and seed (Table 3) -- only
# the --csrd-* flags differ, so a gap is attributable to the objective.
# Usage: scripts/train/train_q8b-1.7b.sh ARM [SEED]
#   sft            B1: LoRA SFT on s1K-Q8B (lambda_r = 0)
#   csrd           default: receiver heads (excess kurtosis, background-subtracted), L_route + L_mass
#   csrd-a         CSRD-A: anchor rows weighted 1 + beta (A8)
#   csrd-c         CSRD-C: causally selected heads + L_causal (needs scripts/targets/causal_heads_qwen3-8b.sh)
#   csrd-pq        CSRD-PQ: per-query, per-head KL (A14)
#   csrd-qk        CSRD-QK: separate Q/K adapter trained only by the routing losses, detached inputs (A12)
#   csrd-nomass    A7: lambda_m = 0 (Lemma 1 predicts this is worse)
#   csrd-band      A5: student routing averaged over the whole band instead of K_S receiver heads
#   csrd-kurtosis  A2: raw-kurtosis heads for teacher and student
#   csrd-causalonly A3: causal target only (L_causal + L_mass, no L_route)
#   csrd-b1/-b2    A4: only the middle / late depth band
# Env: LAMBDA (0.3), LR (1e-4), EPOCHS (3), GPUS ("0"), D_MIN (4), QUERIES (8, A6), LORA_R (64, A13). A trained arm is skipped
# when its output dir already holds adapter_config.json (or config.json for csrd-qk).
set -euo pipefail

ARM="${1:?arm is required (sft, csrd, csrd-a, csrd-c, csrd-pq, csrd-qk, csrd-nomass, csrd-band, csrd-kurtosis, csrd-causalonly, csrd-b1, csrd-b2)}"
SEED="${2:-42}"
read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
for _v in $(compgen -e PET_) $(compgen -e TORCHELASTIC_); do unset "$_v"; done
MASTER_PORT=66$(($RANDOM%90+10))
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --rdzv_backend static --nnodes 1 --node_rank 0 --master_addr localhost --master_port ${MASTER_PORT}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/Qwen3-1.7B-Base"
DATA_PATH="${BASE_PATH}/data/q8b/s1k-student.jsonl"
D_MIN="${D_MIN:-4}"
LAMBDA="${LAMBDA:-0.3}"
LR="${LR:-1.0e-4}"
EPOCHS="${EPOCHS:-3}"
QUERIES="${QUERIES:-8}"
LORA_R="${LORA_R:-64}"                         # A13: 16 | 64 | 128
GLOBAL_BATCH=16
GRAD_ACC=$(( GLOBAL_BATCH / GPUS_PER_NODE ))   # micro-batch 1 per GPU
TARGET_SCORE=excess_bg

CSRD_OPTS=""
case "${ARM}" in
  sft) ;;
  csrd) ;;
  csrd-a) CSRD_OPTS+=" --csrd-anchor-beta 1.0" ;;
  csrd-c) TARGET_SCORE=causal; CSRD_OPTS+=" --causal-dir ${BASE_PATH}/data/q8b/causal" ;;
  csrd-pq) CSRD_OPTS+=" --csrd-loss-form per_query" ;;
  csrd-qk) CSRD_OPTS+=" --csrd-qk-rank ${QK_RANK:-32}" ;;
  csrd-nomass) CSRD_OPTS+=" --csrd-mass-ratio 0" ;;
  csrd-band) CSRD_OPTS+=" --csrd-head-mode band" ;;
  csrd-kurtosis) TARGET_SCORE=kurtosis; CSRD_OPTS+=" --csrd-score kurtosis" ;;
  csrd-causalonly) TARGET_SCORE=causal; CSRD_OPTS+=" --causal-dir ${BASE_PATH}/data/q8b/causal --csrd-route-ratio 0" ;;
  csrd-b1) CSRD_OPTS+=" --csrd-bands 0" ;;
  csrd-b2) CSRD_OPTS+=" --csrd-bands 1" ;;
  *) echo "unknown arm: ${ARM}" >&2; exit 2 ;;
esac

VARIANT=""
[[ "${LORA_R}" != 64 ]] && VARIANT+="-r${LORA_R}"
[[ "${LR}" != 1.0e-4 ]] && VARIANT+="-lr${LR}"
TAG="${ARM}${VARIANT}-q8b-1.7b-s${SEED}"
if [[ "${ARM}" != sft ]]; then
  [[ "${QUERIES}" != 8 ]] && VARIANT+="-m${QUERIES}"
  [[ "${D_MIN}" != 4 ]] && VARIANT+="-d${D_MIN}"
  TAG="${ARM}-l${LAMBDA}${VARIANT}-q8b-1.7b-s${SEED}"
  CSRD_OPTS+=" --csrd-lambda ${LAMBDA} --targets-dir ${BASE_PATH}/data/q8b/targets-dmin${D_MIN}-${TARGET_SCORE}"
  CSRD_OPTS+=" --csrd-d-min ${D_MIN} --csrd-queries ${QUERIES} --csrd-k-student 16"
  CSRD_OPTS+=" --csrd-warmup-frac 0.1 --csrd-ramp-frac 0.1 --csrd-grad-log-interval 20"
fi
OUTPUT_DIR="${BASE_PATH}/checkpoints/${TAG}"
if [[ -f "${OUTPUT_DIR}/adapter_config.json" || -f "${OUTPUT_DIR}/config.json" ]]; then
  echo "skip ${TAG}: ${OUTPUT_DIR} already has a final checkpoint"
  exit 0
fi

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --learning-rate ${LR}"
OPTS+=" --warmup-ratio 0.05"
OPTS+=" --per-device-batch-size 1"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --attn-implementation ${ATTN:-sdpa}"
OPTS+=" --logging-steps 5"
OPTS+=" --save-strategy epoch --save-total-limit 5"
OPTS+=" --seed ${SEED}"
OPTS+=" --lora-r ${LORA_R} --lora-alpha ${LORA_R} --lora-dropout 0.0"
OPTS+=" --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
OPTS+=" --max-seq-len 32768"
OPTS+=" --ce-chunk 4096"
OPTS+=" --gradient-checkpointing"
OPTS+=" --metrics-log ${BASE_PATH}/logs/metrics-${TAG}.json"
OPTS+="${CSRD_OPTS}"
# 1.7B + LoRA fits without ZeRO; DDP keeps every parameter local (the chunked CE reads lm_head.weight
# directly, which ZeRO-3 partitioning would hide). Pass DS_CONFIG for ZeRO-2 if memory is tight.
[[ -n "${DS_CONFIG:-}" ]] && OPTS+=" --deepspeed-config ${DS_CONFIG}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/train_sft.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/train-${TAG}.log"

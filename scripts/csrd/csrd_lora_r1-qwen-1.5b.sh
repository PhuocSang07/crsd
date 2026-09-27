#!/usr/bin/env bash
# Phase 3: CSRD training (read mode) -- DeepSeek-R1-Distill-Qwen-1.5B track, Qwen3-8B teacher signals on s1K-1.1.
# Same config as SGL spectral_lora_r1-qwen-1.5b.sh plus L = L_CE + lambda * L_route + 0.1 * lambda * L_mass.
# Env: GPUS, CSRD_LAMBDA (0.1), CSRD_MASS_RATIO (0.1), CSRD_BANDS ("0,1"; "0" = middle band only), CSRD_D_MIN (4, needs
# the matching signals/q8b-s1k11-dmin<D>-excess_bg.safetensors), CSRD_GRAD_LOG_INTERVAL (20), SEED (42), LR, MIN_LR,
# DS_CONFIG (SGL's ZeRO-2 offload config; "" = plain DDP, e.g. when nvcc is missing).
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
# Offline server: models/data come from the download.txt mirrors; no HF Hub access, no vLLM usage stats.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# ZeRO-2 offload JIT-compiles cpu_adam against system nvcc, which can trail the torch cuXXX
# build -- skip that version check.
export DS_SKIP_CUDA_CHECK=1

# The cluster (PyTorchJob pod) injects PET_RDZV_BACKEND=c10d / PET_RDZV_ENDPOINT=<worker-0>:23456 /
# TORCHELASTIC_*; torchrun reads those over --master_addr and hangs in "Rendezvous'ing worker group"
# waiting on that endpoint. This is a single-node run: drop them and pin the static backend.
for _v in $(compgen -e PET_) $(compgen -e TORCHELASTIC_); do unset "$_v"; done
MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --rdzv_backend static \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: env not found at ${PROJECT_ENV}; build it from crsd.txt (repo root) or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

EFFECTIVE_BATCH=32
if (( EFFECTIVE_BATCH % GPUS_PER_NODE != 0 )); then
  echo "csrd_lora_r1-qwen-1.5b.sh: ${GPUS_PER_NODE} GPUs does not divide effective batch ${EFFECTIVE_BATCH}." >&2
  exit 2
fi

CSRD_LAMBDA="${CSRD_LAMBDA:-0.1}"
CSRD_MASS_RATIO="${CSRD_MASS_RATIO:-0.1}"
CSRD_BANDS="${CSRD_BANDS:-0,1}"
CSRD_D_MIN="${CSRD_D_MIN:-4}"
SEED="${SEED:-42}"
VARIANT=""
[[ "${CSRD_MASS_RATIO}" != 0.1 ]] && VARIANT+="-m${CSRD_MASS_RATIO}"
[[ "${CSRD_BANDS}" == 0 ]] && VARIANT+="-b1"
[[ "${CSRD_BANDS}" == 1 ]] && VARIANT+="-b2"
[[ "${CSRD_D_MIN}" != 4 ]] && VARIANT+="-d${CSRD_D_MIN}"
TAG="csrd-lora-l${CSRD_LAMBDA}${VARIANT}-r1-qwen-1.5b"
[[ "${SEED}" != 42 ]] && TAG+="-s${SEED}"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH="${BASE_PATH}/data/records/s1k11-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl"
SIGNALS_PATH="${BASE_PATH}/signals/q8b-s1k11-dmin${CSRD_D_MIN}-excess_bg.safetensors"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${TAG}"
EPOCHS=3
LR="${LR:-5.0e-5}"
MIN_LR="${MIN_LR:-1.0e-5}"
WARMUP_RATIO=0.1
BATCH_SIZE=1
GRAD_ACC=$(( EFFECTIVE_BATCH / GPUS_PER_NODE ))
ATTN=sdpa
LOG_INTERVAL=5
SAVE_STRATEGY=epoch
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=6
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DS_CONFIG="${DS_CONFIG-${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json}"
MAX_SEQ_LEN=32768

# Student receiver heads are picked after a CE-only warmup (10% of steps); lambda then ramps over the next 10%.
CSRD_CAUSAL_RATIO=0
CSRD_QUERIES=8
CSRD_K_STUDENT=16
CSRD_WARMUP_FRAC=0.1
CSRD_RAMP_FRAC=0.1
CSRD_GRAD_LOG_INTERVAL="${CSRD_GRAD_LOG_INTERVAL:-20}"

[[ -d "${MODEL_NAME}" ]] || { echo "missing local model ${MODEL_NAME} (download.txt)" >&2; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "missing ${DATA_PATH}: run scripts/data/data_r1-qwen-1.5b.sh first" >&2; exit 1; }
[[ -s "${SIGNALS_PATH}" ]] || { echo "missing ${SIGNALS_PATH}: run scripts/teacher/teacher_qwen3-8b.sh (or copy the bank)" >&2; exit 1; }
if [[ -f "${OUTPUT_DIR}/adapter_config.json" ]]; then
  echo "skip ${TAG}: ${OUTPUT_DIR} already has a final adapter"
  exit 0
fi

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --learning-rate ${LR}"
OPTS+=" --min-learning-rate ${MIN_LR}"
OPTS+=" --warmup-ratio ${WARMUP_RATIO}"
OPTS+=" --per-device-batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --attn-implementation ${ATTN}"
OPTS+=" --logging-steps ${LOG_INTERVAL}"
OPTS+=" --save-strategy ${SAVE_STRATEGY}"
OPTS+=" --save-steps ${SAVE_STEPS}"
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --lora-r ${LORA_R}"
OPTS+=" --lora-alpha ${LORA_ALPHA}"
OPTS+=" --lora-dropout ${LORA_DROPOUT}"
OPTS+=" --lora-target-modules ${LORA_TARGET_MODULES}"
[[ -n "${DS_CONFIG}" ]] && OPTS+=" --deepspeed-config ${DS_CONFIG}"
OPTS+=" --max-seq-len ${MAX_SEQ_LEN}"
OPTS+=" --gradient-checkpointing"
OPTS+=" --metrics-log ${BASE_PATH}/logs/metrics-${TAG}.json"
OPTS+=" --signal-bank ${SIGNALS_PATH}"
OPTS+=" --csrd-lambda ${CSRD_LAMBDA}"
OPTS+=" --csrd-mass-ratio ${CSRD_MASS_RATIO}"
OPTS+=" --csrd-causal-ratio ${CSRD_CAUSAL_RATIO}"
OPTS+=" --csrd-d-min ${CSRD_D_MIN}"
OPTS+=" --csrd-bands ${CSRD_BANDS}"
OPTS+=" --csrd-queries ${CSRD_QUERIES}"
OPTS+=" --csrd-k-student ${CSRD_K_STUDENT}"
OPTS+=" --csrd-warmup-frac ${CSRD_WARMUP_FRAC}"
OPTS+=" --csrd-ramp-frac ${CSRD_RAMP_FRAC}"
OPTS+=" --csrd-grad-log-interval ${CSRD_GRAD_LOG_INTERVAL}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/train_sft.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${TAG}.log"

echo ">>> STOP AND READ logs/${TAG}.log: loss_ce should track the SGL SFT run; csrd_zS_b* should approach csrd_zT_b*"
echo ">>> (not overshoot it), and grad_route_ratio should stay ~1-3 after the ramp."

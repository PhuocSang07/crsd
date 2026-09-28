#!/usr/bin/env bash
# MC-CSRD v4 arms (MC_CSRD_proposal_v4.tex Sec. 7.2) -- DeepSeek-R1-Distill-Qwen-1.5B track, Qwen3-8B reader-teacher bank
# with raw mass (signals/...-mc.safetensors), s1K-1.1. Same config as csrd_lora_r1-qwen-1.5b.sh except the objective and
# three v4 conventions shared by every arm: per-band losses averaged (MC_LAMBDA 0.2 = lambda 0.1 of the v3 band sum),
# unbiased query weights, and one fixed student head set (b0's).
#   ARM=b0  CE only (SFT); routing/mass metrics logged; picks the student heads after the warmup and probes there the
#           gradient norm of every objective (csrd-norm-probe.json, b4's lambda)
#   ARM=b1  v3 objective L_route + MASS_RATIO * L_mass
#   ARM=b2  MC-synthetic KL(D_T^syn || D_S^syn), from the cached means
#   ARM=b3  MC-raw KL(D_T || D_S), the proposed method
#   ARM=b4  v3 objective at a lower lambda (the driver passes MC_LAMBDA from the b0 probe)
# Env: ARM, MC_LAMBDA (0.2), MASS_RATIO (0.1), STUDENT_HEADS (b1-b4: b0's heads by default; "" = pick after the warmup),
# CSRD_GRAD_LOG_INTERVAL (10), GPUS, SEED (42), DS_CONFIG (SGL's ZeRO-2 offload; "" = plain DDP).
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
  echo "mc_csrd_lora_r1-qwen-1.5b.sh: ${GPUS_PER_NODE} GPUs does not divide effective batch ${EFFECTIVE_BATCH}." >&2
  exit 2
fi

ARM="${ARM:?ARM=b0|b1|b2|b3|b4}"
MC_LAMBDA="${MC_LAMBDA:-0.2}"
MASS_RATIO="${MASS_RATIO:-0.1}"
SEED="${SEED:-42}"
B0_HEADS="${BASE_PATH}/checkpoints/mc-b0-sft-r1-qwen-1.5b/csrd-student-heads.json"
LAMBDA=${MC_LAMBDA}
PROBE=0
case "${ARM}" in
  b0) OBJECTIVE=none; LAMBDA=0; PROBE=8; TAG=mc-b0-sft; STUDENT_HEADS="" ;;
  b1) OBJECTIVE=route_mass; TAG=mc-b1-route-l${MC_LAMBDA} ;;
  b2) OBJECTIVE=mc_syn; TAG=mc-b2-syn-l${MC_LAMBDA} ;;
  b3) OBJECTIVE=mc_raw; TAG=mc-b3-raw-l${MC_LAMBDA} ;;
  b4) OBJECTIVE=route_mass; TAG=mc-b4-route-l${MC_LAMBDA} ;;
  *) echo "unknown ARM=${ARM} (b0|b1|b2|b3|b4)" >&2; exit 2 ;;
esac
[[ "${OBJECTIVE}" == route_mass && "${MASS_RATIO}" != 0.1 ]] && TAG+="-m${MASS_RATIO}"
TAG+="-r1-qwen-1.5b"
[[ "${SEED}" != 42 ]] && TAG+="-s${SEED}"
STUDENT_HEADS="${STUDENT_HEADS-${B0_HEADS}}"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH="${BASE_PATH}/data/records/s1k11-DeepSeek-R1-Distill-Qwen-1.5B-sgl.jsonl"
SIGNALS_PATH="${BASE_PATH}/signals/q8b-s1k11-dmin4-excess_bg-mc.safetensors"
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

# Student receiver heads: b0 picks them after the CE-only warmup (10% of steps); the other arms load them, so every arm
# is trained and measured on the same heads (their metrics start at step 0). lambda ramps over the next 10%.
CSRD_BANDS=0,1
CSRD_D_MIN=4
CSRD_CAUSAL_RATIO=0
CSRD_QUERIES=8
CSRD_K_STUDENT=16
CSRD_WARMUP_FRAC=0.1
CSRD_RAMP_FRAC=0.1
CSRD_BAND_REDUCTION=mean
CSRD_QUERY_WEIGHTING=unbiased
CSRD_GRAD_LOG_INTERVAL="${CSRD_GRAD_LOG_INTERVAL:-10}"

[[ -d "${MODEL_NAME}" ]] || { echo "missing local model ${MODEL_NAME} (download.txt)" >&2; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "missing ${DATA_PATH}: run scripts/data/data_r1-qwen-1.5b.sh first" >&2; exit 1; }
[[ -s "${SIGNALS_PATH}" ]] || { echo "missing ${SIGNALS_PATH}: run BANK_SUFFIX=-mc scripts/teacher/teacher_qwen3-8b.sh" >&2; exit 1; }
if [[ -n "${STUDENT_HEADS}" && ! -f "${STUDENT_HEADS}" ]]; then
  echo "missing ${STUDENT_HEADS}: train ARM=b0 first (or STUDENT_HEADS=\"\" to pick heads in this run)" >&2
  exit 1
fi
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
OPTS+=" --csrd-objective ${OBJECTIVE}"
OPTS+=" --csrd-lambda ${LAMBDA}"
OPTS+=" --csrd-mass-ratio ${MASS_RATIO}"
OPTS+=" --csrd-causal-ratio ${CSRD_CAUSAL_RATIO}"
OPTS+=" --csrd-d-min ${CSRD_D_MIN}"
OPTS+=" --csrd-bands ${CSRD_BANDS}"
OPTS+=" --csrd-queries ${CSRD_QUERIES}"
OPTS+=" --csrd-k-student ${CSRD_K_STUDENT}"
OPTS+=" --csrd-warmup-frac ${CSRD_WARMUP_FRAC}"
OPTS+=" --csrd-ramp-frac ${CSRD_RAMP_FRAC}"
OPTS+=" --csrd-band-reduction ${CSRD_BAND_REDUCTION}"
OPTS+=" --csrd-query-weighting ${CSRD_QUERY_WEIGHTING}"
OPTS+=" --csrd-probe-microbatches ${PROBE}"
OPTS+=" --csrd-grad-log-interval ${CSRD_GRAD_LOG_INTERVAL}"
if [[ -n "${STUDENT_HEADS}" ]]; then
  OPTS+=" --csrd-head-mode fixed"
  OPTS+=" --csrd-student-heads ${STUDENT_HEADS}"
fi

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/train_sft.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${TAG}.log"

echo ">>> STOP AND READ logs/${TAG}.log: loss_ce should match mc-b0-sft; csrd_ez_b* (|Z_S - Z_T|) and csrd_dM_* (mass gap"
echo ">>> per distance bin) are the v4 calibration metrics, loss_mc_raw = sum_b csrd_mcber_b + csrd_mccond_b (chain rule)."

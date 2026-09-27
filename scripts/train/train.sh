#!/usr/bin/env bash
# Train one arm of a track. Hyperparameters = the baselines' (SpectralGuidedLearning
# project_commands_spectral_r1-qwen-1.5b.sh): LoRA all-linear r = alpha = 16, dropout 0.05, lr 5e-5 cosine to
# 1e-5, warmup 0.1, 3 epochs, effective batch 32 (micro-batch 1), seed 42, 32k tokens, DeepSpeed ZeRO-2 offload,
# on the baselines' exact training text (s1K-1.1, SGL format). Only the --csrd-* flags differ between arms.
# Usage: scripts/train/train.sh TRACK ARM [SEED]
#   sft            LoRA SFT, same engine (the "vanilla" SGL arm)
#   csrd           receiver heads (excess kurtosis, background-subtracted), L_route + L_mass
#   csrd-a         CSRD-A: anchor rows weighted 1 + beta (A8)
#   csrd-c         CSRD-C: causally selected heads + L_causal (needs scripts/targets/causal_heads.sh)
#   csrd-pq        CSRD-PQ: per-query, per-head KL (A14)
#   csrd-qk        CSRD-QK: separate Q/K adapter trained only by the routing losses (A12; DDP, not DeepSpeed)
#   csrd-nomass    A7: lambda_m = 0              csrd-band      A5: whole band instead of K_S receiver heads
#   csrd-kurtosis  A2: raw-kurtosis heads        csrd-causalonly A3: L_causal + L_mass, no L_route
#   csrd-b1/-b2    A4: middle / late band only
# Env: LAMBDA (0.3), LR (5e-5), EPOCHS (3), LORA_R (16, A13), QUERIES (8, A6), D_MIN (4, A1), GPUS.
# A trained arm is skipped when its output dir already holds adapter_config.json (config.json for csrd-qk).
set -euo pipefail
ARM="${2:?arm is required (sft, csrd, csrd-a, csrd-c, csrd-pq, csrd-qk, csrd-nomass, csrd-band, csrd-kurtosis, csrd-causalonly, csrd-b1, csrd-b2)}"
SEED="${3:-42}"
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "${1:-}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export DS_SKIP_CUDA_CHECK=1
for _v in $(compgen -e PET_) $(compgen -e TORCHELASTIC_); do unset "$_v"; done
GPUS_PER_NODE=${#GPUS[@]}
EFFECTIVE_BATCH=32
(( EFFECTIVE_BATCH % GPUS_PER_NODE == 0 )) || { echo "${GPUS_PER_NODE} GPUs do not divide batch ${EFFECTIVE_BATCH}" >&2; exit 2; }
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --rdzv_backend static --nnodes 1 --node_rank 0 --master_addr localhost --master_port 66$(($RANDOM%90+10))"

LAMBDA="${LAMBDA:-0.3}"
LR="${LR:-5.0e-5}"
MIN_LR="${MIN_LR:-1.0e-5}"
EPOCHS="${EPOCHS:-3}"
LORA_R="${LORA_R:-16}"
QUERIES="${QUERIES:-8}"
DS_CONFIG="${DS_CONFIG-${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json}"
BANK="${SIGNALS}"

# L_causal belongs to CSRD-C only (Sec. 4.9): the bank also carries causal targets, so every other arm sets lambda_c = 0.
CSRD_OPTS=""
[[ "${ARM}" != csrd-c && "${ARM}" != csrd-causalonly ]] && CSRD_OPTS+=" --csrd-causal-ratio 0"
case "${ARM}" in
  sft|csrd) ;;
  csrd-a) CSRD_OPTS+=" --csrd-anchor-beta 1.0" ;;
  csrd-c) BANK="signals/${TT}-${TRAIN_CANON}${SEG_TAG}-dmin${D_MIN}-causal.safetensors" ;;
  csrd-causalonly) BANK="signals/${TT}-${TRAIN_CANON}${SEG_TAG}-dmin${D_MIN}-causal.safetensors"; CSRD_OPTS+=" --csrd-route-ratio 0" ;;
  csrd-pq) CSRD_OPTS+=" --csrd-loss-form per_query" ;;
  csrd-qk) CSRD_OPTS+=" --csrd-qk-rank ${QK_RANK:-32}"; DS_CONFIG="" ;;  # tensor-hook gradient routing needs DDP
  csrd-nomass) CSRD_OPTS+=" --csrd-mass-ratio 0" ;;
  csrd-band) CSRD_OPTS+=" --csrd-head-mode band" ;;
  csrd-kurtosis) BANK="signals/${TT}-${TRAIN_CANON}${SEG_TAG}-dmin${D_MIN}-kurtosis.safetensors"; CSRD_OPTS+=" --csrd-score kurtosis" ;;
  csrd-b1) CSRD_OPTS+=" --csrd-bands 0" ;;
  csrd-b2) CSRD_OPTS+=" --csrd-bands 1" ;;
  *) echo "unknown arm: ${ARM}" >&2; exit 2 ;;
esac

VARIANT=""
[[ "${LORA_R}" != 16 ]] && VARIANT+="-r${LORA_R}"
[[ "${LR}" != 5.0e-5 ]] && VARIANT+="-lr${LR}"
[[ -n "${SEG_TAG}" ]] && VARIANT+="${SEG_TAG}"
if [[ "${ARM}" == sft ]]; then
  TAG="sft${VARIANT}-${TRACK}-s${SEED}"
else
  [[ "${QUERIES}" != 8 ]] && VARIANT+="-m${QUERIES}"
  [[ "${D_MIN}" != 4 ]] && VARIANT+="-d${D_MIN}"
  TAG="${ARM}-l${LAMBDA}${VARIANT}-${TRACK}-s${SEED}"
  [[ -s "${BANK}" ]] || { echo "missing signal bank ${BANK} (scripts/targets/teacher_signals.sh ${TRACK})" >&2; exit 2; }
  CSRD_OPTS+=" --csrd-lambda ${LAMBDA} --signals ${BANK} --csrd-d-min ${D_MIN} --csrd-queries ${QUERIES} --csrd-k-student 16"
  CSRD_OPTS+=" --csrd-warmup-frac 0.1 --csrd-ramp-frac 0.1 --csrd-grad-log-interval 20"
fi
OUTPUT_DIR="checkpoints/${TAG}"
if [[ -f "${OUTPUT_DIR}/adapter_config.json" || -f "${OUTPUT_DIR}/config.json" ]]; then
  echo "skip ${TAG}: ${OUTPUT_DIR} already has a final checkpoint"; exit 0
fi

OPTS=""
OPTS+=" --model-name ${STUDENT}"
OPTS+=" --data-path ${STUDENT_TRAIN_RECORDS}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --learning-rate ${LR}"
OPTS+=" --min-learning-rate ${MIN_LR}"
OPTS+=" --warmup-ratio 0.1"
OPTS+=" --per-device-batch-size 1"
OPTS+=" --gradient-accumulation-steps $(( EFFECTIVE_BATCH / GPUS_PER_NODE ))"
OPTS+=" --attn-implementation sdpa"
OPTS+=" --logging-steps 5"
OPTS+=" --save-strategy epoch --save-total-limit 6"
OPTS+=" --seed ${SEED}"
OPTS+=" --lora-r ${LORA_R} --lora-alpha ${LORA_R} --lora-dropout 0.05"
OPTS+=" --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
OPTS+=" --max-seq-len 32768"
OPTS+=" --gradient-checkpointing"
OPTS+=" --metrics-log logs/metrics-${TAG}.json"
[[ -n "${DS_CONFIG}" ]] && OPTS+=" --deepspeed-config ${DS_CONFIG}"
OPTS+="${CSRD_OPTS}"

CMD="torchrun ${DISTRIBUTED_ARGS} src/train_sft.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "logs/train-${TAG}.log"

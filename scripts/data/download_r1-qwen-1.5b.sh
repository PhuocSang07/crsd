#!/usr/bin/env bash
# Phase 0 (ONLINE, once per new server): mirror the models / datasets listed in download.txt into LOCAL_MODELS_ROOT /
# LOCAL_DATA_ROOT, the same layout the offline infra builds from download.txt. Everything after this phase runs with
# HF_HUB_OFFLINE=1. Not for the audited B200 (its infra downloads from download.txt itself).
#   LOCAL_MODELS_ROOT=/data/models LOCAL_DATA_ROOT=/data/datasets bash scripts/data/download_r1-qwen-1.5b.sh
# ALL=true also fetches the lines below the "ignored" marker (other tracks / diagnostics). HF_TOKEN is used when set.
# DRY_RUN=true only prints what would be downloaded where.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_PATH}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: env not found at ${PROJECT_ENV}; run scripts/setup.sh first (or set PROJECT_ENV)" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
DOWNLOAD_LIST="${BASE_PATH}/download.txt"
ALL="${ALL:-false}"

download() {  # download {model|dataset} REPO_ID LOCAL_DIR
  local kind=$1 repo=$2 dir=$3
  if [[ -d "${dir}" ]] && [[ -n "$(ls -A "${dir}" 2>/dev/null)" ]]; then
    echo "skip ${repo} (${dir} exists)"; return
  fi
  echo ">>> ${repo} (${kind}) -> ${dir}"
  [[ "${DRY_RUN:-false}" == true ]] && return
  python - "${kind}" "${repo}" "${dir}" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
kind, repo, local_dir = sys.argv[1:]
snapshot_download(repo_id=repo, repo_type=kind, local_dir=local_dir, token=os.environ.get("HF_TOKEN"))
PY
}

ignored=false
while read -r line; do
  [[ "${line}" == "(this line and below are ignored)"* ]] && { ignored=true; continue; }
  [[ "${ignored}" == true && "${ALL}" != true ]] && break
  read -r flag repo dir _ <<< "${line}"
  case "${flag}" in
    --hf-dataset) download dataset "${repo}" "${dir/\/mnt\/local\/_data\/@PROJECT@/${LOCAL_DATA_ROOT}}" ;;
    --hf)         download model "${repo}" "${dir/\/mnt\/local\/_models\/@PROJECT@/${LOCAL_MODELS_ROOT}}" ;;
    *) ;;
  esac
done < "${DOWNLOAD_LIST}"

echo ">>> done: models under ${LOCAL_MODELS_ROOT}, datasets under ${LOCAL_DATA_ROOT}"

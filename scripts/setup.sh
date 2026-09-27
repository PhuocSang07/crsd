#!/usr/bin/env bash
# Install the full stack (training + eval/vLLM) into the CSRD venv via uv, pyproject.toml.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-false}"

command -v uv >/dev/null || { echo "ERROR: uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2; exit 1; }
# torch 2.13 cu130 wheels (Blackwell B200 and Hopper H100/H200 alike) need an NVIDIA driver with CUDA 13.0 (>= 580).
if command -v nvidia-smi >/dev/null; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | driver ${DRIVER}"
  (( ${DRIVER%%.*} >= 580 )) || echo "WARN: driver ${DRIVER} < 580 cannot run the cu130 wheels; point [[tool.uv.index]] in pyproject.toml at an older CUDA build" >&2
fi
# Same torch/vllm/transformers pins as SpectralGuidedLearning/ (cu130 index in pyproject.toml; pins = crsd.txt).
UV_PROJECT_ENVIRONMENT="${PROJECT_ENV}" uv sync
VENV_PY="${PROJECT_ENV}/bin/python"

[[ "${INSTALL_FLASH_ATTN}" == true ]] && uv pip install --python "${VENV_PY}" flash-attn --no-build-isolation

"${VENV_PY}" - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
from vllm import LLM
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda} | vllm import OK")
PY
echo "setup done"

#!/usr/bin/env bash
# Install the full stack (training + eval/vLLM) into the CSRD venv via uv, pyproject.toml.
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/crsd}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-false}"

command -v uv >/dev/null || { echo "ERROR: uv not found" >&2; exit 1; }
# Same torch/vllm/transformers pins as SpectralGuidedLearning/ (cu130 index in pyproject.toml).
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

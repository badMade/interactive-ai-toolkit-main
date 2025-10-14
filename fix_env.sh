#!/usr/bin/env bash
set -euo pipefail

# Ensure script runs from project root
if [[ ! -f "run.py" ]]; then
  echo "Error: fix_env.sh must be executed from the project root containing run.py." >&2
  exit 1
fi

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [[ "${OS_NAME}" != "Darwin" || "${ARCH_NAME}" != "x86_64" ]]; then
  echo "Error: This environment fix script only supports macOS on x86_64." >&2
  echo "Detected platform: ${OS_NAME} ${ARCH_NAME}" >&2
  exit 1
fi

if [[ -d ".venv" ]]; then
  echo "Removing existing virtual environment at .venv"
  rm -rf .venv
fi

echo "Creating new Python 3.12 virtual environment"
python3.12 -m venv .venv

source .venv/bin/activate
export PIP_REQUIRE_VIRTUALENV=1

python -m pip install --upgrade pip

python -m pip install --force-reinstall \
  numpy==1.26.4 \
  torch==2.2.2 \
  torchvision==0.17.2 \
  torchaudio==2.2.2 \
  openai-whisper==20250625 \
  transformers==4.57.0 \
  soundfile==0.13.1 \
  sentencepiece==0.2.1 \
  pytest==8.4.2 \
  imageio[ffmpeg]

python - <<'PY'
import numpy
import torch
import torchvision
import torchaudio
import transformers
import soundfile
import sentencepiece
import whisper
import imageio
import imageio_ffmpeg

print("Environment setup verification succeeded.")
PY

#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[fix_env] ERROR: Setup aborted. See messages above." >&2' ERR

log() {
  printf '[fix_env] %s\n' "$*"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

log "Verifying platform ..."
uname_s=$(uname -s)
if [ "$uname_s" != "Darwin" ]; then
  printf '[fix_env] Unsupported operating system: %s. This script must run on macOS.\n' "$uname_s" >&2
  exit 1
fi

uname_m=$(uname -m)
if [ "$uname_m" != "x86_64" ]; then
  printf '[fix_env] Unsupported architecture: %s. This script targets macOS x86_64.\n' "$uname_m" >&2
  exit 1
fi

if [ ! -f "run.py" ]; then
  printf '[fix_env] run.py not found in %s. Please run this script from the project root.\n' "$PWD" >&2
  exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  cat >&2 <<'EOF_INNER'
[fix_env] Error: python3.12 not found in PATH.
Install it with Homebrew:
  brew install python@3.12
EOF_INNER
  exit 1
fi

log "Creating a fresh Python 3.12 virtual environment ..."
if [ -d ".venv" ] || [ -L ".venv" ]; then
  log "Removing existing .venv directory ..."
  rm -rf .venv
fi
python3.12 -m venv .venv

if [ ! -f ".venv/bin/activate" ]; then
  printf '[fix_env] Virtual environment creation failed at %s/.venv\n' "$PWD" >&2
  exit 1
fi

log "Activating virtual environment ..."
set +u
. .venv/bin/activate
set -u

export PIP_REQUIRE_VIRTUALENV=1

log "Upgrading pip ..."
python -m pip install --upgrade pip

log "Writing pinned requirements.txt ..."
cat <<'EOF_REQ' > requirements.txt
numpy==1.26.4
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
openai-whisper==20250625
transformers==4.57.0
soundfile==0.13.1
sentencepiece==0.2.1
pytest==8.4.2
EOF_REQ

log "Installing dependencies (force reinstall) ..."
python -m pip install --upgrade --force-reinstall -r requirements.txt

log "Running sanity import check ..."
python - <<'PY'
import numpy, torch, whisper
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("Whisper: OK")
PY

log "Environment setup complete."

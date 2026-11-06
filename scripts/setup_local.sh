#!/usr/bin/env bash
# Local setup for Linux and macOS: virtual environment, dependencies, local folders, checks.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.10 or newer (see README section 2)."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10):
    sys.exit("ERROR: Python 3.10 or newer is required, found %d.%d" % sys.version_info[:2])
PY

if ! command -v git >/dev/null 2>&1; then
  echo "WARNING: git not found. Install it before cloning or pushing (see README section 2)."
fi

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data/uit_hwdb data/corpus checkpoints/ocr checkpoints/corrector outputs

echo "Running unit tests ..."
python -m unittest discover -s tests -t .

echo "Validating the sample contract file ..."
python scripts/validate_ocr_out.py contracts/examples/ocr_out.sample.jsonl

echo
echo "Setup finished. To activate the environment in a new terminal:"
echo "  bash or zsh : source .venv/bin/activate"
echo "  fish        : source .venv/bin/activate.fish"

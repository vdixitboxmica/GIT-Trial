#!/usr/bin/env bash
set -euo pipefail

if [ -d "venv" ]; then VENV_DIR="venv";
elif [ -d ".venv" ]; then VENV_DIR=".venv";
else VENV_DIR=".venv"; fi
echo "Using virtual environment at: ${VENV_DIR}"

[ -f "${VENV_DIR}/bin/activate" ] || python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
[ -f requirements.txt ] && pip install -r requirements.txt
[ -f requirements-dev.txt ] && pip install -r requirements-dev.txt
if [ -f "pyproject.toml" ] && grep -q "^\[project\]" pyproject.toml; then
  pip install -e .
fi

python -V
echo "✅ Dev env ready"

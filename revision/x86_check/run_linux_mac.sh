#!/usr/bin/env bash
# Linux/macOS: the same three steps as the .bat files. Usage: bash run_linux_mac.sh
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
python run_check.py check
python run_check.py run all
python run_check.py summary

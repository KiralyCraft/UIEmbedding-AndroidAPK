#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--docker" ]]; then
    shift
    docker build -f tools/Dockerfile.export -t ui-f6-export .
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/workspace" ui-f6-export "$@"
else
    PYTHON="${EXPORT_PYTHON:-python3.11}"
    if [[ ! -x .venv-export/bin/python ]]; then
        command -v "$PYTHON" >/dev/null || { echo 'Python 3.11 required. Alternatively use: ./00_export_model.sh --docker' >&2; exit 1; }
        "$PYTHON" -m venv .venv-export
    fi
    .venv-export/bin/python -m pip install --upgrade pip
    .venv-export/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tools/export-requirements.txt
    .venv-export/bin/python tools/export_f6.py "$@"
fi

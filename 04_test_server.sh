#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${SERVER_TEST_PYTHON:-python3}" -m venv .venv-server-tests
.venv-server-tests/bin/python -m pip install -r server/requirements.txt -r server/requirements-test.txt
mkdir -p reports
cd server
PYTHONPATH=. ../.venv-server-tests/bin/python -m pytest -q --junitxml=../reports/server-tests.xml tests

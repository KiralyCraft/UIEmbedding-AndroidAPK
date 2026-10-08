#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .env ]] || { echo 'Copy .env.example to .env and set random database passwords first.' >&2; exit 1; }
python3 tools/check_assets.py
python3 tools/check_env.py
mkdir -p exports
docker compose build api
docker compose up -d --wait mysql
docker compose run --rm api python -m collector.cli init-db
docker compose run --rm api python -m collector.cli register-model /models/f6_manifest.json
docker compose up -d api
echo 'API is bound to 127.0.0.1:8000. Configure the HTTPS reverse proxy and create a user with the CLI.'

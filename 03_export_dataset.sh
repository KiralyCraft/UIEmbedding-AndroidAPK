#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $# -ge 1 ]] || { echo "Usage: $0 USERNAME [PACKAGE]" >&2; exit 1; }
mkdir -p exports
DEST="/exports/export-$(date -u +%Y%m%dT%H%M%SZ)"
ARGS=(--username "$1" --output "$DEST")
if [[ $# -ge 2 ]]; then ARGS+=(--package "$2"); fi
# Bind exports to the caller's UID/GID instead of creating inaccessible root-owned files.
docker compose exec --user "$(id -u):$(id -g)" api python -m collector.cli export "${ARGS[@]}"
echo "Export: .${DEST}"

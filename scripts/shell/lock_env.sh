#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-artifacts/runs/unknown}"
mkdir -p "$OUT_DIR"
# Export a hashed requirements mirror from uv.lock (optional)
uv export --format requirements-txt --strict --output-file "$OUT_DIR/requirements.lock.txt"

#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:-/data}"
OUT_DIR="${2:-artifacts/runs/unknown}"
mkdir -p "$OUT_DIR"
LC_ALL=C find "$DATA_DIR" -type f -print0 | xargs -0 sha256sum > "$OUT_DIR/data.sha256"

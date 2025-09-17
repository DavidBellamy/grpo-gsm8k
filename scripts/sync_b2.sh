#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?RUN_DIR required}"
REMOTE="${2:-b2:my-bucket/grpo-gsm8k}"
# Expect Rclone config via env or /root/.config/rclone/rclone.conf
rclone copy -P "$RUN_DIR" "$REMOTE/$(basename "$RUN_DIR")"

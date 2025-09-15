#!/usr/bin/env bash
set -euo pipefail

# --- Deterministic env (yours) ---
[ -f /etc/profile.d/env.det.sh ] && . /etc/profile.d/env.det.sh || true

# Ensure workspace exists (your volume usually mounts here)
mkdir -p /workspace

# --- Persistent caches & VS Code server on your Network Volume ---
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/workspace/.config}"
export VSCODE_AGENT_FOLDER="${VSCODE_AGENT_FOLDER:-/workspace/.vscode-server}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
mkdir -p "$XDG_CONFIG_HOME" "$VSCODE_AGENT_FOLDER" /workspace/bin

# Make the CLI available persistently
export PATH="/workspace/bin:${PATH}"

# --- VS Code CLI (tunnels) — install once, reuse forever ---
if ! command -v code >/dev/null 2>&1; then
  echo "[entrypoint] Installing VS Code CLI..."
  if ! curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-x64" \
      | tar -xz --strip-components=1 -C /workspace/bin */bin/code; then
    echo "[entrypoint] VS Code CLI install failed; continuing without it."
  fi
fi

# Autostart tunnel if requested (set VSCODE_TUNNEL_NAME in the pod template)
if [ -n "${VSCODE_TUNNEL_NAME:-}" ] && command -v code >/dev/null 2>&1; then
  ( nohup code tunnel --name "$VSCODE_TUNNEL_NAME" \
      --accept-server-license-terms --disable-telemetry \
      >/workspace/.code-tunnel.log 2>&1 & ) || true
fi

# --- Python env: prefer persistent venv on the volume; fallback to baked /opt/venv ---
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/workspace/.venv}"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "[entrypoint] Creating persistent venv at $UV_PROJECT_ENVIRONMENT"
  uv venv "$UV_PROJECT_ENVIRONMENT"
  # If a project exists in /workspace, hydrate deps (non-frozen for dev iteration)
  if [ -f /workspace/pyproject.toml ]; then
    echo "[entrypoint] Syncing Python deps with uv (dev mode)"
    uv sync || true
  fi
fi

# Activate the chosen environment; fallback to your baked /opt/venv
if [ -x "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$UV_PROJECT_ENVIRONMENT/bin/activate"
elif [ -x "/opt/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . /opt/venv/bin/activate
fi

# Work from the mounted volume by default
cd /workspace

# Hand off to the actual command (tini will reap as PID 1)
exec "$@"

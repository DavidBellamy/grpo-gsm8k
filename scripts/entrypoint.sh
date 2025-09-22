#!/usr/bin/env bash
# /usr/local/bin/entrypoint.sh
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*"; }

# --- Deterministic env (yours) ---
[ -f /etc/profile.d/env.det.sh ] && . /etc/profile.d/env.det.sh || true

# Ensure workspace exists (your volume usually mounts here)
mkdir -p /workspace

# Ensure DATA_DIR exists if set (for dataset snapshots, etc.)
[ -n "${DATA_DIR:-}" ] && mkdir -p "$DATA_DIR"

# --- Persistent caches & VS Code server/CLI on your Network Volume ---
# These ENV are also set in the Dockerfile so all processes see them. We ensure dirs exist here.
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/workspace/.config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"

export VSCODE_CLI_DATA_DIR="${VSCODE_CLI_DATA_DIR:-/workspace/.vscode-cli}"
export VSCODE_CLI_USE_FILE_KEYCHAIN="${VSCODE_CLI_USE_FILE_KEYCHAIN:-1}"
export VSCODE_AGENT_FOLDER="${VSCODE_AGENT_FOLDER:-/workspace/.vscode-server}"

mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$PIP_CACHE_DIR"
mkdir -p "$VSCODE_CLI_DATA_DIR" "$VSCODE_AGENT_FOLDER" /workspace/bin

# Keep /workspace/bin on PATH for ALL future shells (login + interactive)
if [ ! -f /etc/profile.d/workspace-path.sh ]; then
  cat >/etc/profile.d/workspace-path.sh <<'SH'
# ensure /workspace/bin is on PATH even for login shells
case ":$PATH:" in *":/workspace/bin:"*) ;; *) export PATH="/workspace/bin:$PATH" ;; esac
SH
fi
# Also cover interactive non-login bash shells (VS Code terminals, etc.)
if [ -f /etc/bash.bashrc ] && ! grep -q '/workspace/bin' /etc/bash.bashrc; then
  echo 'case ":$PATH:" in *":/workspace/bin:"*) ;; *) export PATH="/workspace/bin:$PATH";; esac' >> /etc/bash.bashrc
fi
# Make the CLI available in this process immediately
case ":$PATH:" in *":/workspace/bin:"*) ;; *) export PATH="/workspace/bin:$PATH";; esac

# --- VS Code CLI fallback install (image already installs it; this is a backup) ---
if ! command -v code >/dev/null 2>&1; then
  log "VS Code CLI not found, attempting runtime install..."
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  for os in cli-linux-x64 cli-alpine-x64; do
    url="https://update.code.visualstudio.com/latest/${os}/stable"
    log "Fetching: $url"
    if curl -fsSL -o "$tmp/cli.tgz" "$url" && tar -tzf "$tmp/cli.tgz" >/dev/null 2>&1; then
      tar -xzf "$tmp/cli.tgz" -C "$tmp"
      bin_path="$(find "$tmp" -type f -name code -perm -u+x | head -n1 || true)"
      if [ -n "${bin_path:-}" ]; then
        install -m 0755 "$bin_path" /workspace/bin/code
        ln -sfn /workspace/bin/code /usr/local/bin/code
        log "VS Code CLI installed: $(/workspace/bin/code --version | head -n1)"
        break
      fi
    fi
  done
fi

# --- VS Code Tunnel autostart (if requested) ---
vscode_tunnel_connected() {
  # Try JSON first (if CLI supports it), fall back to plain text grep.
  if code tunnel status --format json >/dev/null 2>&1; then
    local state
    state="$(code tunnel status --format json 2>/dev/null | jq -r '.tunnel.state // .tunnel.tunnel // empty')"
    [ "$state" = "Connected" ]
  else
    code tunnel status 2>/dev/null | grep -q 'Connected'
  fi
}

start_vscode_tunnel() {
  local name="${VSCODE_TUNNEL_NAME:-runpod-$(hostname | cut -c1-20)}"
  log "Starting VS Code tunnel: name=${name}"
  nohup code tunnel \
    --name "${name}" \
    --accept-server-license-terms \
    --disable-telemetry \
    --no-sleep \
    >>/workspace/.code-tunnel.log 2>&1 &
}

if [ -n "${VSCODE_TUNNEL_NAME:-}" ] && command -v code >/dev/null 2>&1; then
  if vscode_tunnel_connected; then
    log "VS Code tunnel already connected; not starting another."
  else
    # If a stale process exists, cleanly stop it so new flags take effect (no-op if none).
    code tunnel kill >/dev/null 2>&1 || true
    start_vscode_tunnel
  fi
fi

# --- Python env: prefer persistent venv on the volume; fallback to baked /opt/venv ---
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/workspace/.venv}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  log "Creating persistent venv at $UV_PROJECT_ENVIRONMENT"
  uv venv "$UV_PROJECT_ENVIRONMENT"
  # If a project exists in /workspace, hydrate deps
  if [ -f /workspace/pyproject.toml ]; then
    if [ -f /workspace/uv.lock ]; then
      log "Syncing deps from uv.lock (frozen, no-dev)"
      uv sync --frozen --no-dev || true
    else
      log "Syncing deps from pyproject (no-dev)"
      uv sync --no-dev || true
    fi
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

# Hand off to the actual command
exec "$@"

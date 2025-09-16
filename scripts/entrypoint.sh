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

# Make the CLI available persistently (current process)
export PATH="/workspace/bin:${PATH}"

# --- VS Code CLI (tunnels) — install once, reuse forever ---
install_vscode_cli() {
  echo "[entrypoint] Installing VS Code CLI..."
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  # Try glibc build first, then musl (Alpine)
  for os in cli-linux-x64 cli-alpine-x64; do
    url="https://code.visualstudio.com/sha/download?build=stable&os=${os}"
    echo "[entrypoint] Fetching: $url"
    if curl -fL --retry 3 --retry-delay 2 --retry-connrefused -o "$tmp/cli.tgz" "$url"; then
      # Quick sanity check: is it a gzip tar?
      if tar -tzf "$tmp/cli.tgz" >/dev/null 2>&1; then
        tar -xzf "$tmp/cli.tgz" -C "$tmp"
        # Find the 'code' binary anywhere in the archive
        bin_path="$(find "$tmp" -type f -name code -perm -u+x | head -n1 || true)"
        if [ -n "${bin_path:-}" ]; then
          install -m 0755 "$bin_path" /workspace/bin/code
          echo "[entrypoint] VS Code CLI installed: $(/workspace/bin/code --version | head -n1)"
          return 0
        fi
      fi
    fi
  done

  echo "[entrypoint] VS Code CLI install failed; continuing without it."
  return 1
}

if ! command -v code >/dev/null 2>&1; then
  install_vscode_cli || true
fi

# Belt-and-suspenders: expose code via standard PATH dir too
if [ -x /workspace/bin/code ]; then
  ln -sfn /workspace/bin/code /usr/local/bin/code
fi

# Persist VS Code CLI state (auth/server) on the volume
mkdir -p /workspace/.vscode
ln -sfn /workspace/.vscode /root/.vscode

# Autostart tunnel if requested (set VSCODE_TUNNEL_NAME in the pod template)
if [ -n "${VSCODE_TUNNEL_NAME:-}" ] && command -v code >/dev/null 2>&1; then
  # 2) If not authenticated yet, kick off device login once (code + URL goes to the log)
  if ! code tunnel status 2>/dev/null | grep -q '"tunnel":"Connected"'; then
    if ! pgrep -f "code.*tunnel user login" >/dev/null 2>&1; then
      ( nohup code tunnel user login >/workspace/.code-tunnel.log 2>&1 & ) || true
    fi
  fi

  # 3) Start exactly one tunnel (background) and log output
  if ! pgrep -f "code.*tunnel.*--name[ =]${VSCODE_TUNNEL_NAME}" >/dev/null 2>&1; then
    ( nohup code tunnel --name "$VSCODE_TUNNEL_NAME" \
        --accept-server-license-terms --disable-telemetry \
        >/workspace/.code-tunnel.log 2>&1 & ) || true
  fi
fi

# --- Python env: prefer persistent venv on the volume; fallback to baked /opt/venv ---
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/workspace/.venv}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"   # make sure venv bins are on PATH even before activate

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "[entrypoint] Creating persistent venv at $UV_PROJECT_ENVIRONMENT"
  uv venv "$UV_PROJECT_ENVIRONMENT"
  # If a project exists in /workspace, hydrate deps (non-frozen for dev iteration)
  if [ -f /workspace/pyproject.toml ]; then
    if [ -f /workspace/uv.lock ]; then
      echo "[entrypoint] Syncing deps from uv.lock (frozen, no-dev)"
      uv sync --frozen --no-dev || true
    else
      echo "[entrypoint] Syncing deps from pyproject (no-dev)"
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

# syntax=docker/dockerfile:1.7

# H100-friendly, reproducible base (CUDA 12.8 + Ubuntu 22.04)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
# Pin apt to a snapshot for reproducible OS packages (override at build with --build-arg)
ARG UBUNTU_SNAPSHOT=20240927T000000Z

# Repoint apt to snapshot.ubuntu.com and install minimal runtime deps
RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
  set -eux; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" > /etc/apt/sources.list; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy-updates main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" >> /etc/apt/sources.list; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy-security main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" >> /etc/apt/sources.list; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    ca-certificates curl git jq unzip \
    bash bash-completion tzdata \
    tini tmux htop \
    # Python runtime must exist in final image so entrypoint can create/use /workspace/.venv
    python3.10 python3.10-venv python3-pip; \
  ln -snf /usr/share/zoneinfo/UTC /etc/localtime && echo UTC > /etc/timezone; \
  rm -rf /var/lib/apt/lists/*

# Install uv/uvx (single static binaries)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /usr/local/bin/uvx

# Workspace and env defaults
WORKDIR /workspace
ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    # Prefer prebuilt wheels where possible
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    # Put baked venv first at runtime
    PATH="/opt/venv/bin:/workspace/.venv/bin:${PATH}" \
    # Performance-friendly defaults; flip at runtime if you need strict determinism
    TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=0 \
    NVIDIA_TF32_OVERRIDE=1 \
    # VS Code tunnel persistence (optional)
    VSCODE_CLI_DATA_DIR=/workspace/.vscode-cli \
    VSCODE_CLI_USE_FILE_KEYCHAIN=1 \
    VSCODE_AGENT_FOLDER=/workspace/.vscode-server

# (Optional) Install VS Code CLI at build time; pin via --build-arg VSCODE_VERSION=<ver>
ARG VSCODE_CLI_OS=cli-linux-x64
ARG VSCODE_VERSION=latest
RUN set -eux; \
  url="https://update.code.visualstudio.com/${VSCODE_VERSION}/${VSCODE_CLI_OS}/stable"; \
  curl -fsSL "$url" -o /tmp/cli.tgz; \
  tar -xzf /tmp/cli.tgz -C /tmp; \
  install -m 0755 "$(find /tmp -type f -name code -perm -u+x | head -n1)" /usr/local/bin/code; \
  rm -rf /tmp/cli.tgz /tmp/*

# Place helpers (no source code baked)
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/env.det.sh /etc/profile.d/env.det.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /etc/profile.d/env.det.sh

# =========================
# Deps stage: bake /opt/venv
# =========================
FROM base AS deps

# Toolchain for building Python wheels (vLLM, etc.) lives only in deps stage
RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
  set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 python3.10-venv python3-pip python3.10-dev; \
  rm -rf /var/lib/apt/lists/*

# Only metadata to trigger rebuild when deps change
COPY uv.lock pyproject.toml /workspace/

# Create baked runtime venv and install only runtime deps
RUN uv venv /opt/venv && \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:${PATH}" uv sync --frozen --no-dev

# =========================
# App stage: final image
# =========================
FROM base AS app

# Reuse prebuilt environment from deps
COPY --from=deps /opt/venv /opt/venv

# Final entrypoint/cmd live ONLY here
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]

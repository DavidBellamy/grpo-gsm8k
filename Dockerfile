# H100-friendly, reproducible base (CUDA 12.8 + Ubuntu 22.04)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# YYYYMMDDThhmmssZ from https://snapshot.ubuntu.com/
ARG UBUNTU_SNAPSHOT=20250701T000000Z

# ---- Lock apt to a snapshot for reproducible OS packages ----
RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt \
  set -eux; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" > /etc/apt/sources.list; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy-updates main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" >> /etc/apt/sources.list; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy-security main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" >> /etc/apt/sources.list; \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential tzdata ca-certificates curl wget unzip git jq bash-completion \
    tmux htop \
    python3.10 python3.10-venv python3-pip python3.10-dev; \
  ln -snf /usr/share/zoneinfo/UTC /etc/localtime && echo UTC > /etc/timezone;

ENV TZ=UTC \
  PYTHONUNBUFFERED=1 \
  # toolchain hint for Triton/vLLM
  CC=gcc \
  CXX=g++

# ---- Install uv (fast package manager) ----
# Official guidance: copy the uv binary from the uv image
# https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /usr/local/bin/uvx

# ---- Reproducibility- & performance-friendly uv defaults ----
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    PATH="/workspace/.venv/bin:${PATH}" \
    DATA_DIR=/workspace/data

# ---- Runtime defaults for (fast) determinism knobs ----
ENV TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=0 \
    # Fast math by default (flip at run-time if you want strict determinism)
    NVIDIA_TF32_OVERRIDE=1

# ---- VS Code tunnel persistence (auth & server live on /workspace) ----
ENV VSCODE_CLI_DATA_DIR=/workspace/.vscode-cli \
    VSCODE_CLI_USE_FILE_KEYCHAIN=1 \
    VSCODE_AGENT_FOLDER=/workspace/.vscode-server

# ---- (Optional) Install VS Code CLI at build time ----
# Pin with VSCODE_VERSION=<version or commit> or leave as "latest"
ARG VSCODE_CLI_OS=cli-linux-x64
ARG VSCODE_VERSION=latest
RUN set -eux; \
  url="https://update.code.visualstudio.com/${VSCODE_VERSION}/${VSCODE_CLI_OS}/stable"; \
  curl -fsSL "$url" -o /tmp/cli.tgz; \
  tar -xzf /tmp/cli.tgz -C /tmp; \
  install -m 0755 "$(find /tmp -type f -name code -perm -u+x | head -n1)" /usr/local/bin/code; \
  rm -rf /tmp/cli.tgz /tmp/*

# ---- Place helpers (no source code baked) ----
WORKDIR /workspace
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/env.det.sh /etc/profile.d/env.det.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /etc/profile.d/env.det.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]

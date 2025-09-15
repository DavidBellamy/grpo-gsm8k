# H100-friendly, reproducible base (CUDA 12.8 + Ubuntu 22.04)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# YYYYMMDDThhmmssZ from https://snapshot.ubuntu.com/
ARG UBUNTU_SNAPSHOT=20250701T000000Z

# ---- Lock apt to a snapshot for reproducible OS packages ----
RUN set -eux; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" > /etc/apt/sources.list; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy-updates main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" >> /etc/apt/sources.list; \
  printf 'deb http://snapshot.ubuntu.com/ubuntu/%s jammy-security main restricted universe multiverse\n' "$UBUNTU_SNAPSHOT" >> /etc/apt/sources.list; \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl wget unzip git jq bash-completion \
    tmux htop \
    python3.10 python3.10-venv python3-pip python3.10-dev build-essential pkg-config \
    tini; \
  ln -snf /usr/share/zoneinfo/UTC /etc/localtime && echo UTC > /etc/timezone; \
  rm -rf /var/lib/apt/lists/*

ENV TZ=UTC \
    PYTHONUNBUFFERED=1

# ---- Install uv (fast package manager) ----
# Official guidance: copy the uv binary from the uv image
# https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /usr/local/bin/uvx

# ---- Reproducibility- & performance-friendly uv defaults ----
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# ---- Preinstall heavy deps via uv using your lockfile ----
# Copy only pyproject/lock so Docker layer-caches deps installs.
# (We do NOT copy source code; it will come from the mounted volume.)
WORKDIR /image-deps
COPY pyproject.toml uv.lock ./
# If you rely on the PyTorch cu128 index, pyproject must declare it; see below.
RUN python3.10 -m venv /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---- Runtime defaults for (fast) determinism knobs ----
ENV TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=0 \
    # Fast math by default (flip at run-time if you want strict determinism)
    NVIDIA_TF32_OVERRIDE=1

# ---- Place helpers (no source code baked) ----
WORKDIR /workspace
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/env.det.sh /etc/profile.d/env.det.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /etc/profile.d/env.det.sh

# Let tini reap orphans; keep a clean PID 1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

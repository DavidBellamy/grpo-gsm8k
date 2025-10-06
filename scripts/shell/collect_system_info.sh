#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-artifacts/runs/unknown}"
mkdir -p "$OUT_DIR/sys"

# Set all timestamps to UTC
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT_DIR/sys/time_utc.txt"
python - << 'PY' > "$OUT_DIR/sys/time.json"
import json, time, datetime
print(json.dumps({
  "utc_iso": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
  "epoch_s": time.time(),
  "epoch_ns": time.time_ns()
}, indent=2))
PY

# GPU + driver + topology
{ nvidia-smi -L || true; nvidia-smi --query-gpu=name,driver_version,compute_cap,pcie.link.width.max,pcie.link.gen.max --format=csv,noheader || true; } > "$OUT_DIR/sys/nvidia_smi_basic.txt" 2>&1
{ nvidia-smi -q || true; } > "$OUT_DIR/sys/nvidia_smi_q.txt" 2>&1
{ nvidia-smi topo -m || true; } > "$OUT_DIR/sys/topo.txt" 2>&1
{ nvidia-smi nvlink --status || true; } > "$OUT_DIR/sys/nvlink.txt" 2>&1
{ nvidia-smi -q -d MIG || true; } > "$OUT_DIR/sys/mig.txt" 2>&1

# NCCL version (best-effort)
python - << 'PY' > "$OUT_DIR/sys/nccl.json" || true
import json, torch, os, sys
v = None
try:
    v = torch.cuda.nccl.version()
except Exception:
    pass
print(json.dumps({"torch_nccl_version": v}, indent=2))
PY

# CUDA libs (best-effort strings)
( ldconfig -p | grep -i -E 'cudnn|cublas' || true ) > "$OUT_DIR/sys/cuda_libs_ldconfig.txt" 2>&1
( for so in /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda*/targets/x86_64-linux/lib/libcublas*.so*; do \
      [ -f "$so" ] && (echo "== $so" ; strings "$so" | grep -m1 -E 'CUDNN|CUBLAS|Version' || true); \
  done ) > "$OUT_DIR/sys/cuda_libs_strings.txt" 2>&1

# Python & packages
python -V > "$OUT_DIR/sys/python_version.txt"
pip --version > "$OUT_DIR/sys/pip_version.txt"
pip freeze --all > "$OUT_DIR/sys/pip.freeze.txt"

# Environment flags of interest
env | sort > "$OUT_DIR/sys/env.txt"

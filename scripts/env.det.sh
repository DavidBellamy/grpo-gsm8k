# Loaded by /etc/profile.d and by run scripts
export TZ=UTC
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
# Stricter cuBLAS determinism: (PyTorch detects it)
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
# Make sure TF32 is off unless you explicitly enable it
export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-0}
# Optional: tame thread noise a bit
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

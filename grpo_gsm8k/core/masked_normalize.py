import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum tensor values where mask == 1/True and divide by a constant.

    Args:
        tensor: The input tensor.
        mask: Boolean or {0,1} tensor of the same shape as `tensor`. Positions
              with 1/True are included in the sum.
        normalize_constant: The constant to divide the (masked) sum by.
        dim: Dimension along which to sum. If None, sum over all dimensions.

    Returns:
        The normalized masked sum. If dim is None, returns a scalar tensor.
        Otherwise returns a tensor with `dim` reduced (like torch.sum).
    """
    if tensor.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} must match tensor shape {tensor.shape}")

    # Convert mask to the same dtype as tensor to preserve gradients in tensor
    mask_f = mask.to(dtype=tensor.dtype)
    masked = tensor * mask_f

    norm_const = torch.as_tensor(normalize_constant, dtype=tensor.dtype, device=tensor.device)
    if dim is None:
        summed = masked.sum()
    else:
        summed = masked.sum(dim=dim)

    return summed / norm_const

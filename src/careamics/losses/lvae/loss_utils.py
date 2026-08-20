import torch


def free_bits_kl(
    kl: torch.Tensor, free_bits: float, batch_average: bool = False, eps: float = 1e-6
) -> torch.Tensor:
    """Compute free-bits version of KL divergence.

    This function ensures that the KL doesn't go to zero for any latent dimension.
    Hence, it contributes to use latent variables more efficiently, leading to
    better representation learning.

    NOTE:
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).

    Parameters
    ----------
    kl : torch.Tensor
        The KL divergence tensor with shape (batch size, layers).
    free_bits : float
        The free bits value. Set to 0.0 to disable free bits.
    batch_average : bool
        Whether to average over the batch before clamping to `free_bits`.
    eps : float
        A small value to avoid numerical instability.

    Returns
    -------
    torch.Tensor
        The free-bits version of the KL divergence with shape (layers,).
    """
    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)

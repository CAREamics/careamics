import torch


def free_bits_kl(
    kl: torch.Tensor, free_bits: float, batch_average: bool = False, eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes free-bits version of KL divergence.

    Ensures that the KL doesn't go to zero for any latent dimension.
    Hence, it contributes to use latent variables more efficiently,
    leading to better representation learning.

    NOTE:
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).

    Args:
        kl (torch.Tensor)
        free_bits (float)
        batch_average (bool, optional))
        eps (float, optional)

    Returns
    -------
        The KL with free bits
    """
    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)


def get_kl_weight(kl_annealing, kl_start, kl_annealtime, kl_weight, current_epoch):
    """
    KL loss can be weighted depending whether any annealing procedure is used.

    This function computes the weight of the KL loss in case of annealing.
    """
    if kl_annealing:
        # calculate relative weight
        kl_weight = (current_epoch - kl_start) * (1.0 / kl_annealtime)
        # clamp to [0,1]
        kl_weight = min(max(0.0, kl_weight), 1.0)

        # if the final weight is given, then apply that weight on top of it
        if kl_weight is not None:
            kl_weight = kl_weight * kl_weight
    elif kl_weight is not None:
        return kl_weight
    else:
        kl_weight = 1.0
    return kl_weight

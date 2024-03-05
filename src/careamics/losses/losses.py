"""
Loss submodule.

This submodule contains the various losses used in CAREamics.
"""

import torch


def n2v_loss(
    samples: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, device: str
) -> torch.Tensor:
    """
    N2V Loss function (see Eq.7 in Krull et al).

    Parameters
    ----------
    samples : torch.Tensor
        Patches with manipulated pixels.
    labels : torch.Tensor
        Noisy patches.
    masks : torch.Tensor
        Array containing masked pixel locations.
    device : str
        Device to use.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    errors = (labels - samples) ** 2
    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss

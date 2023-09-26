import torch


def n2v_loss(
    samples: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, device: str
) -> torch.Tensor:
    """The loss function as described in Eq. 7 of the paper.

    Parameters
    ----------
    samples : torch.Tensor
        patches with masked pixels
    labels : torch.Tensor
        noisy patches
    masks : torch.Tensor
        array containing masked pixel locations
    device : str
        The device to use

    Returns
    -------
    torch.Tensor
        loss
    """
    samples, labels, masks = samples.to(device), labels.to(device), masks.to(device)
    errors = (labels - samples) ** 2

    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss

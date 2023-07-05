import torch


def n2v_loss(samples, labels, masks, device):
    """The loss function as described in Eq. 7 of the paper."""
    samples, labels, masks = samples.to(device), labels.to(device), masks.to(device)
    errors = (labels - samples) ** 2

    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss


def pn2v_loss(samples, labels, masks, noiseModel):
    """The loss function as described in Eq. 7 of the paper."""
    likelihoods = noiseModel.likelihood(labels, samples)
    likelihoods_avg = torch.log(torch.mean(likelihoods, dim=0, keepdim=True)[0, ...])

    # Average over pixels and batch
    loss = -torch.sum(likelihoods_avg * masks) / torch.sum(masks)
    return loss

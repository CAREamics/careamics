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


def decon_loss(
    samples, labels, masks, std, psf, regularization, positivity_constraint, device
):
    from .loss_utils import tgv3

    # TODO refactor!
    # psf = artificial_psf(size_of_psf=81, std_gauss=sigma).to('cuda')

    psf_shape = psf.shape[2]
    pad_size = (psf_shape - 1) // 2
    assert psf.shape[2] == psf.shape[3]
    rs = torch.mean(samples, dim=0).reshape(
        samples.shape[1], 1, samples.shape[2], samples.shape[3]
    )
    # we pad the result
    rs = torch.nn.functional.pad(
        rs, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )

    # surrogate
    base = torch.full(labels.shape, 0.5).to(device)
    base_pad = torch.nn.functional.pad(
        base, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )
    base_conv = (
        torch.nn.functional.conv2d(base_pad, weight=psf, padding=0, stride=1) + 1e-12
    )
    rel_blur = labels / base_conv
    blur_pad = torch.nn.functional.pad(
        rel_blur, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )
    incr = torch.nn.functional.conv2d(blur_pad, weight=psf, padding=0, stride=1)
    base *= incr

    # and convolve it with the psf
    conv = torch.nn.functional.conv2d(rs, weight=psf, padding=0, stride=[1, 1])
    conv = conv.reshape(samples.shape[1], samples.shape[2], samples.shape[3])

    # This is the implementation of the positivity constraint

    signs = (samples < 0).float()
    samples_positivity_constraint = samples * signs

    # the N2V loss
    errors = (labels - conv) ** 2
    loss = torch.sum(errors * masks) / torch.sum(masks)

    # TV regularization

    # reg = tv_regularization(samples)
    # r2 = tgvk_reg(samples, k=1)

    reg = tgv3(samples, device)

    return (
        loss / (std**2)
        + (
            reg * regularization
            + positivity_constraint
            * torch.mean(torch.abs(samples_positivity_constraint))
        )
        / std,
        reg * regularization / std,
    )

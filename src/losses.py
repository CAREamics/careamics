import torch
import numpy as np


def n2v_loss(samples, labels, masks, std=None):
    """
    The loss function as described in Eq. 7 of the paper.
    """

    errors = (labels - torch.mean(samples, dim=0)) ** 2

    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss / (std**2)


def pn2v_loss(samples, labels, masks, noiseModel):
    """
    The loss function as described in Eq. 7 of the paper.
    """

    likelihoods = noiseModel.likelihood(labels, samples)
    likelihoods_avg = torch.log(torch.mean(likelihoods, dim=0, keepdim=True)[0, ...])

    # Average over pixels and batch
    loss = -torch.sum(likelihoods_avg * masks) / torch.sum(masks)
    return loss


def tgvk_reg(samples: torch.Tensor, k: int) -> torch.Tensor:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    samples : torch.Tensor
        Input image
    k : int
        TGV order, max is image size

    Returns
    -------
    torch.Tensor
        _description_
    """

    assert samples.shape[-2] == samples.shape[-1]

    tgv = 0

    for i in range(2, min(k + 2, samples.shape[-1])):
        print("I", i)
        yprior = (samples[:, :, i:, 1:-1] - samples[:, :, :-i, 1:-1]) / 2.0
        xprior = (samples[:, :, 1:-1, i:] - samples[:, :, 1:-1, :-i]) / 2.0
        tgv += torch.mean(torch.sqrt(yprior**2 + xprior**2 + 1e-15))

    return tgv


def anorm(x):
    """Calculate L2 norm over the last array dimention"""
    return torch.sqrt((x * x).sum(-1) + 1e-15)


def nabla(I, device):
    b, c, h, w = I.shape
    G = torch.zeros((b, c, h, w, 2), dtype=I.dtype).to(device)
    G[:, :, :, :-1, 0] -= I[:, :, :, :-1]
    G[:, :, :, :-1, 0] += I[:, :, :, 1:]
    G[:, :, :-1, :, 1] -= I[:, :, :-1, :]
    G[:, :, :-1, :, 1] += I[:, :, 1:, :]
    return G


def epsilon(I, device):
    b, c, h, w, _ = I.shape
    G = torch.zeros((b, c, h, w, 4), dtype=I.dtype).to(device)
    G[:, :, :, :-1, 0] -= I[:, :, :, :-1, 0]  # xdx
    G[:, :, :, :-1, 0] += I[:, :, :, 1:, 0]
    G[:, :, :-1, :, 1] -= I[:, :, :-1, :, 0]  # xdy
    G[:, :, :-1, :, 1] += I[:, :, 1:, :, 0]
    G[:, :, :, :-1, 2] -= I[:, :, :, :-1, 1]  # ydx
    G[:, :, :, :-1, 2] += I[:, :, :, 1:, 1]
    G[:, :, :-1, :, 3] -= I[:, :, :-1, :, 1]  # ydy
    G[:, :, :-1, :, 3] += I[:, :, 1:, :, 1]
    return G


def eps3(I, device):
    b, c, h, w, _ = I.shape
    G = torch.zeros((b, c, h, w, 8), dtype=I.dtype).to(device)
    G[:, :, :, :-1, 0] -= I[:, :, :, :-1, 0]  # xdx
    G[:, :, :, :-1, 0] += I[:, :, :, 1:, 0]
    G[:, :, :-1, :, 1] -= I[:, :, :-1, :, 0]  # xdy
    G[:, :, :-1, :, 1] += I[:, :, 1:, :, 0]

    G[:, :, :, :-1, 2] -= I[:, :, :, :-1, 1]  # ydx
    G[:, :, :, :-1, 2] += I[:, :, :, 1:, 1]
    G[:, :, :-1, :, 3] -= I[:, :, :-1, :, 1]  # ydy
    G[:, :, :-1, :, 3] += I[:, :, 1:, :, 1]

    G[:, :, :, :-1, 4] -= I[:, :, :, :-1, 2]  # xdx
    G[:, :, :, :-1, 4] += I[:, :, :, 1:, 2]
    G[:, :, :-1, :, 5] -= I[:, :, :-1, :, 2]  # xdy
    G[:, :, :-1, :, 5] += I[:, :, 1:, :, 2]

    G[:, :, :, :-1, 6] -= I[:, :, :, :-1, 3]  # ydx
    G[:, :, :, :-1, 6] += I[:, :, :, 1:, 3]
    G[:, :, :-1, :, 7] -= I[:, :, :-1, :, 3]  # ydy
    G[:, :, :-1, :, 7] += I[:, :, 1:, :, 3]
    return G


def tgv(samples, device):
    energy_tgv = anorm(epsilon(nabla(samples, device), device)).mean()
    return energy_tgv


def tgv3(samples, device):
    energy_tgv = anorm(eps3(epsilon(nabla(samples, device), device), device)).mean()
    return energy_tgv


def tv_regularization(samples):
    yprior = (samples[:, :, 2:, 1:-1] - samples[:, :, :-2, 1:-1]) / 2.0  # ** 2
    xprior = (samples[:, :, 1:-1, 2:] - samples[:, :, 1:-1, :-2]) / 2.0  # ** 2

    # g = torch.tensor([[-1, 0, 1],
    #                 [-2, 0, 2],
    #                 [-1, 0, 1]]).reshape(1, 1, 3, 3).to(torch.float32)

    # g2 = torch.tensor([[0, -1, 0],
    #                 [-1, 4, -1],
    #                 [0, -1, 0]]).reshape(1, 1, 3, 3).to(torch.float32)

    # g2_2 = torch.tensor([[1, -2, 1],
    #                 [2, -4, 2],
    #                 [1, -2, 1]]).reshape(1, 1, 3, 3).to(torch.float32)

    # g2_2d = torch.tensor([[1, 0, -1],
    #                 [0, 0, 0],
    #                 [-1, 0, 1]]).reshape(1, 1, 3, 3).to(torch.float32)

    # gl = torch.tensor([[-1, -1, -1],
    #                 [-1, 8, -1],
    #                 [-1, -1, -1]]).reshape(1, 1, 3, 3).to(torch.float32)

    # yprior_c = torch.nn.functional.conv2d(samples,
    #                                 weight=g.cuda(),
    #                                 padding=0,
    #                                 stride=[1,1])

    # xprior_c = torch.nn.functional.conv2d(samples,
    #                                 weight=g.transpose(2, 3).cuda(),
    #                                 padding=0,
    #                                 stride=[1,1])

    # return torch.mean(torch.sqrt(n_prior**2 + 1e-15))
    # return torch.mean(torch.sqrt((yprior_c2)**2 + (dprior_c2)**2 + 1e-15))
    # return torch.mean(torch.sqrt(xyprior_gl**2 + 1e-15))

    return torch.mean(torch.sqrt(yprior**2 + xprior**2 + 1e-15))


def decon_loss(
    samples, labels, masks, std, psf, regularization, positivity_constraint, device
):
    #TODO refactor! 
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

"""COSDD model utilities.

Ported from the COSDD reference implementation
(https://github.com/krulllab/COSDD, `models/utils.py`), MIT License,
Copyright (c) 2023 krulllab.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, Normal


class Rotate90(nn.Module):
    """Rotate an input tensor and its conditioning tensor by 90 degrees.

    Used by the autoregressive decoder to align the noise correlation axis with the
    last spatial axis.

    Parameters
    ----------
    k : int
        Number of times to rotate by 90 degrees.
    dims : Sequence of int
        Plane of rotation.
    """

    def __init__(self, k: int, dims: Sequence[int]) -> None:
        """Constructor.

        Parameters
        ----------
        k : int
            Number of times to rotate by 90 degrees.
        dims : Sequence of int
            Plane of rotation.
        """
        super().__init__()
        self.k = k
        self.dims = list(dims)

    def forward(
        self, x: torch.Tensor, s_code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate both tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        s_code : torch.Tensor
            Conditioning tensor.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Rotated input and conditioning tensors.
        """
        x = torch.rot90(x, k=self.k, dims=self.dims)
        s_code = torch.rot90(s_code, k=self.k, dims=self.dims)
        return x, s_code


def sample_mixture_model(
    logweights: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Sample from a Gaussian mixture model.

    Parameters
    ----------
    logweights : torch.Tensor
        Log mixture weights, mixture components along the last axis.
    loc : torch.Tensor
        Component means, mixture components along the last axis.
    scale : torch.Tensor
        Component standard deviations, mixture components along the last axis.

    Returns
    -------
    torch.Tensor
        Samples, with the mixture axis removed.
    """
    px = MixtureSameFamily(Categorical(logits=logweights), Normal(loc, scale))
    return px.sample()


def interleave(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Interleave two tensors along the channel dimension.

    Parameters
    ----------
    tensor1 : torch.Tensor
        The first input tensor.
    tensor2 : torch.Tensor
        The second input tensor.

    Returns
    -------
    torch.Tensor
        The interleaved tensor.

    Raises
    ------
    ValueError
        If the shapes of tensor1 and tensor2 do not match.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors must have the same shape, got {tensor1.shape} and "
            f"{tensor2.shape}."
        )

    result = torch.stack((tensor1, tensor2), dim=2)

    N, C = result.shape[:2]
    result = result.reshape(N, 2 * C, *result.shape[3:])

    return result


def get_padded_size(size: Sequence[int], n_downsc: Sequence[int]) -> list[int]:
    """Calculate the padded size of an image for a number of downscaling steps.

    Parameters
    ----------
    size : Sequence of int
        Spatial size of the image.
    n_downsc : Sequence of int
        Number of downscaling steps along each spatial axis.

    Returns
    -------
    list of int
        The padded size of the image, divisible by `2**n_downsc` along each axis.
    """
    dwnsc = [2**d for d in n_downsc]
    padded_size = [((s - 1) // d + 1) * d for s, d in zip(size, dwnsc, strict=True)]

    return padded_size


def spatial_pad_crop(x: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
    """Pad or crop the input tensor `x` to match the target spatial size.

    Padding and cropping are applied symmetrically.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be padded or cropped, shape (N, C, *spatial).
    target_size : Sequence of int
        The target spatial size.

    Returns
    -------
    torch.Tensor
        The padded or cropped tensor.
    """
    x_size = x.size()[2:]
    delta = [ts - xs for ts, xs in zip(target_size, x_size, strict=True)]
    crop_delta = [
        (abs(d) // 2, abs(d) // 2 + abs(d) % 2) if d < 0 else (0, 0) for d in delta
    ]
    pad_delta = [(d // 2, d // 2 + d % 2) if d > 0 else (0, 0) for d in delta]

    pad = []
    for d in reversed(pad_delta):
        pad.append(d[0])
        pad.append(d[1])
    x = nn.functional.pad(x, pad)
    x_size = x.size()[2:]
    crop = [slice(0, x.size(0)), slice(0, x.size(1))]
    crop += [slice(d[0], xs - d[1]) for d, xs in zip(crop_delta, x_size, strict=True)]
    return x[tuple(crop)]


class LinearUpsample(nn.Module):
    """Upsample using linear, bilinear or trilinear interpolation.

    Parameters
    ----------
    scale_factor : Sequence of int
        The scale factor along each spatial axis.
    """

    def __init__(self, scale_factor: Sequence[int]) -> None:
        """Constructor.

        Parameters
        ----------
        scale_factor : Sequence of int
            The scale factor along each spatial axis.
        """
        super().__init__()
        self.scale_factor = list(scale_factor)
        self.modes = {1: "linear", 2: "bilinear", 3: "trilinear"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the input.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor, shape (N, C, *spatial).

        Returns
        -------
        torch.Tensor
            The upsampled tensor.
        """
        d = x.dim() - 2
        mode = self.modes[d]
        return nn.functional.interpolate(
            x, scale_factor=[float(s) for s in self.scale_factor], mode=mode
        )

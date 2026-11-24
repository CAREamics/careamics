"""Script for utility functions needed by the LVAE model."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.distributions.normal import Normal


def torch_nanmean(inp: torch.Tensor) -> torch.Tensor:
    """Compute the mean of a tensor ignoring NaN values.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor, possibly containing NaN values.

    Returns
    -------
    torch.Tensor
        The mean of the non-NaN elements of the input.
    """
    return torch.mean(inp[~inp.isnan()])


class Enum:
    """Lightweight enum-like base class backed by class attributes."""

    @classmethod
    def name(cls, enum_type: int) -> str | None:
        """Return the attribute name matching the given value.

        Parameters
        ----------
        enum_type : int
            The value to look up.

        Returns
        -------
        str or None
            The name of the attribute holding the given value, or `None` if not found.
        """
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key
        return None

    @classmethod
    def contains(cls, enum_type: int) -> bool:
        """Return whether the given value is defined in the enum.

        Parameters
        ----------
        enum_type : int
            The value to look up.

        Returns
        -------
        bool
            Whether the value is defined in the enum.
        """
        for _key, value in cls.__dict__.items():
            if enum_type == value:
                return True
        return False

    @classmethod
    def from_name(cls, enum_type_str: str) -> int:
        """Return the value for the given attribute name.

        Parameters
        ----------
        enum_type_str : str
            The name of the attribute to look up.

        Returns
        -------
        int
            The value held by the attribute.
        """
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        raise ValueError(f"{cls.__name__}:{enum_type_str} does not exist.")


class LossType(Enum):
    """Enumeration of the loss types supported by the LVAE training code."""

    Elbo = 0
    ElboWithCritic = 1
    ElboMixedReconstruction = 2
    MSE = 3
    ElboWithNbrConsistency = 4
    ElboSemiSupMixedReconstruction = 5
    ElboCL = 6
    ElboRestrictedReconstruction = 7
    DenoiSplitMuSplit = 8


def _pad_crop_img(
    x: torch.Tensor, size: Sequence[int], mode: Literal["crop", "pad"]
) -> torch.Tensor:
    """Pad or crop a tensor of shape (B, C, [Z], Y, X) to a new spatial shape.

    Parameters
    ----------
    x : torch.Tensor
        Input image of shape (B, C, [Z], Y, X).
    size : Sequence[int]
        Desired spatial size ([Z*], Y*, X*).
    mode : Literal["crop", "pad"]
        Whether to 'pad' or 'crop' the input.

    Returns
    -------
    torch.Tensor
        The padded or cropped tensor.
    """
    # TODO: Support cropping/padding on selected dimensions
    assert (x.dim() == 4 and len(size) == 2) or (x.dim() == 5 and len(size) == 3)

    size = tuple(size)
    x_size = x.size()[2:]

    if mode == "pad":
        cond = any(x_size[i] > size[i] for i in range(len(size)))
    elif mode == "crop":
        cond = any(x_size[i] < size[i] for i in range(len(size)))

    if cond:
        raise ValueError(f"Trying to {mode} from size {x_size} to size {size}")

    diffs = [abs(x - s) for x, s in zip(x_size, size, strict=False)]
    d1 = [d // 2 for d in diffs]
    d2 = [d - (d // 2) for d in diffs]

    if mode == "pad":
        if x.dim() == 4:
            padding = [d1[1], d2[1], d1[0], d2[0], 0, 0, 0, 0]
        elif x.dim() == 5:
            padding = [d1[2], d2[2], d1[1], d2[1], d1[0], d2[0], 0, 0, 0, 0]
        return nn.functional.pad(x, padding)
    elif mode == "crop":
        if x.dim() == 4:
            return x[:, :, d1[0] : (x_size[0] - d2[0]), d1[1] : (x_size[1] - d2[1])]
        elif x.dim() == 5:
            return x[
                :,
                :,
                d1[0] : (x_size[0] - d2[0]),
                d1[1] : (x_size[1] - d2[1]),
                d1[2] : (x_size[2] - d2[2]),
            ]


def pad_img_tensor(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Pad a tensor of shape (B, C, [Z], Y, X) to the desired spatial dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input image of shape (B, C, [Z], Y, X).
    size : Sequence[int]
        Desired spatial size ([Z*], Y*, X*).

    Returns
    -------
    torch.Tensor
        The padded tensor.
    """
    return _pad_crop_img(x, size, "pad")


def crop_img_tensor(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Crop a tensor of shape (B, C, [Z], Y, X) to the desired spatial dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input image of shape (B, C, [Z], Y, X).
    size : Sequence[int]
        Desired spatial size ([Z*], Y*, X*).

    Returns
    -------
    torch.Tensor
        The cropped tensor.
    """
    return _pad_crop_img(x, size, "crop")


class StableExponential:
    """Numerically stable redefinition of ``exp()`` and its inverse ``log()``.

    The definitions of exp() and log() are redefined to increase numerical stability,
    while remaining one the inverse of the other (``x = log(exp(x))`` and
    ``x = exp(log(x))`` always hold).

    Definition::

        exp(x) = { exp(x) if x <= 0 ; x + 1    if x > 0 }
        log(x) = { x       if x <= 0 ; log(1+x) if x > 0 }

    NOTE 1:
        Everything is done on the tensor given as input to the constructor. Therefore,
        when exp() is called, ``self._tensor.exp()`` is computed; when log() is called,
        ``torch.log(self._tensor.exp())`` is computed instead.

    NOTE 2:
        Given the output from exp(), ``torch.log()`` or the log() method of the class
        give identical results.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor on which the stable operations are performed.
    """

    def __init__(self, tensor: torch.Tensor):
        """Constructor.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor on which the stable operations are performed.
        """
        self._raw_tensor = tensor
        posneg_dic = self.posneg_separation(self._raw_tensor)
        self.pos_f, self.neg_f = posneg_dic["filter"]
        self.pos_data, self.neg_data = posneg_dic["value"]

    def posneg_separation(self, tensor: torch.Tensor) -> dict:
        """Split a tensor into its positive and non-positive parts.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to split.

        Returns
        -------
        dict
            A dictionary with the positive/negative boolean masks under ``"filter"``
            and the clipped positive/negative tensors under ``"value"``.
        """
        pos = tensor > 0
        pos_tensor = torch.clip(tensor, min=0)

        neg = tensor <= 0
        neg_tensor = torch.clip(tensor, max=0)

        return {"filter": [pos, neg], "value": [pos_tensor, neg_tensor]}

    def exp(self) -> torch.Tensor:
        """Compute the numerically stable exponential of the tensor.

        Returns
        -------
        torch.Tensor
            The stable exponential of the input tensor.
        """
        return torch.exp(self.neg_data) * self.neg_f + (1 + self.pos_data) * self.pos_f

    def log(self) -> torch.Tensor:
        """Compute the numerically stable logarithm of the tensor.

        Returns
        -------
        torch.Tensor
            The stable logarithm of the input tensor.
        """
        return self.neg_data * self.neg_f + torch.log(1 + self.pos_data) * self.pos_f


class StableLogVar:
    """Numerically stable implementation of Log-Variance.

    It relies on the exp() and log() formulas defined in the `StableExponential` class.

    Parameters
    ----------
    logvar : torch.Tensor
        The input (true) logvar vector, to be converted in the stable version.
    enable_stable : bool, optional
        Whether to compute the stable version of log-variance. Default is `True`.
    var_eps : float, optional
        The minimum value attainable by the variance. Default is `1e-6`.
    """

    def __init__(
        self, logvar: torch.Tensor, enable_stable: bool = True, var_eps: float = 1e-6
    ):
        """Constructor.

        Parameters
        ----------
        logvar : torch.Tensor
            The input (true) logvar vector, to be converted in the stable version.
        enable_stable : bool, optional
            Whether to compute the stable version of log-variance. Default is `True`.
        var_eps : float, optional
            The minimum value attainable by the variance. Default is `1e-6`.
        """
        self._lv = logvar
        self._enable_stable = enable_stable
        self._eps = var_eps

    def get(self) -> torch.Tensor:
        """Return the (possibly stabilized) log-variance.

        Returns
        -------
        torch.Tensor
            The log-variance tensor.
        """
        if self._enable_stable is False:
            return self._lv

        return torch.log(self.get_var())

    def get_var(self) -> torch.Tensor:
        """Compute the variance from the log-variance.

        Returns
        -------
        torch.Tensor
            The variance tensor.
        """
        if self._enable_stable is False:
            return torch.exp(self._lv)
        return StableExponential(self._lv).exp() + self._eps

    def get_std(self) -> torch.Tensor:
        """Compute the standard deviation from the log-variance.

        Returns
        -------
        torch.Tensor
            The standard-deviation tensor.
        """
        return torch.sqrt(self.get_var())

    @property
    def is_3D(self) -> bool:
        """Check if the log-variance tensor is 3D.

        Recall that, in this framework, tensors have shape (B, C, [Z], Y, X).

        Returns
        -------
        bool
            Whether the tensor is 3D (i.e. has 5 dimensions).
        """
        return self._lv.dim() == 5

    def centercrop_to_size(self, size: Sequence[int]) -> None:
        """Centercrop the log-variance tensor to the desired size.

        Parameters
        ----------
        size : Sequence[int]
            The desired size of the log-variance tensor.
        """
        assert not self.is_3D, "Centercrop is implemented only for 2D tensors."

        if self._lv.shape[-1] == size:
            return

        diff = self._lv.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._lv = F.center_crop(self._lv, (size, size))


class StableMean:
    """Thin wrapper around a mean tensor exposing stable-distribution helpers.

    Parameters
    ----------
    mean : torch.Tensor
        The mean tensor to wrap.
    """

    def __init__(self, mean: torch.Tensor):
        """Constructor.

        Parameters
        ----------
        mean : torch.Tensor
            The mean tensor to wrap.
        """
        self._mean = mean

    def get(self) -> torch.Tensor:
        """Return the wrapped mean tensor.

        Returns
        -------
        torch.Tensor
            The mean tensor.
        """
        return self._mean

    @property
    def is_3D(self) -> bool:
        """Check if the mean tensor is 3D.

        Recall that, in this framework, tensors have shape (B, C, [Z], Y, X).

        Returns
        -------
        bool
            Whether the tensor is 3D (i.e. has 5 dimensions).
        """
        return self._mean.dim() == 5

    def centercrop_to_size(self, size: Sequence[int]) -> None:
        """Centercrop the mean tensor to the desired size.

        Implemented only in the case of 2D tensors.

        Parameters
        ----------
        size : Sequence[int]
            The desired size of the mean tensor.
        """
        assert not self.is_3D, "Centercrop is implemented only for 2D tensors."

        if self._mean.shape[-1] == size:
            return

        diff = self._mean.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._mean = F.center_crop(self._mean, (size, size))


def allow_numpy(func):
    """Wrap a function so that numpy-array positional arguments are cast to tensors.

    Optional (keyword) arguments are passed through unchanged; positional arguments that
    are numpy arrays are converted to torch tensors before calling the wrapped function.

    Parameters
    ----------
    func : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function.
    """

    def numpy_wrapper(*args, **kwargs):
        """Cast numpy-array positional arguments to tensors, then call ``func``.

        Parameters
        ----------
        *args : Any
            Positional arguments; numpy arrays are converted to tensors.
        **kwargs : Any
            Keyword arguments, passed through unchanged.

        Returns
        -------
        Any
            The output of the wrapped function.
        """
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.Tensor(arg)
            new_args.append(arg)
        new_args = tuple(new_args)

        output = func(*new_args, **kwargs)
        return output

    return numpy_wrapper


class Interpolate(nn.Module):
    """Wrapper for ``torch.nn.functional.interpolate``.

    Parameters
    ----------
    size : int or tuple of int, optional
        The target output size. Exactly one of `size` and `scale` must be given.
    scale : float, optional
        The spatial scale factor. Exactly one of `size` and `scale` must be given.
    mode : str, optional
        The interpolation mode. Default is ``"bilinear"``.
    align_corners : bool, optional
        The ``align_corners`` flag passed to ``interpolate``. Default is `False`.
    """

    def __init__(self, size=None, scale=None, mode="bilinear", align_corners=False):
        """Constructor.

        Parameters
        ----------
        size : int or tuple of int, optional
            The target output size. Exactly one of `size` and `scale` must be given.
        scale : float, optional
            The spatial scale factor. Exactly one of `size` and `scale` must be given.
        mode : str, optional
            The interpolation mode. Default is ``"bilinear"``.
        align_corners : bool, optional
            The ``align_corners`` flag passed to ``interpolate``. Default is `False`.
        """
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to interpolate.

        Returns
        -------
        torch.Tensor
            The interpolated tensor.
        """
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return out


def kl_normal_mc(z, p_mulv, q_mulv):
    """Estimate the element-wise KL between two diagonal multivariate normals.

    One-sample Monte-Carlo estimation, working for any number of dimensions, with
    broadcasting supported (be careful).

    Parameters
    ----------
    z : torch.Tensor
        The sample at which the KL is estimated.
    p_mulv : tuple
        The (mean, log-variance) wrappers of the prior distribution ``p``.
    q_mulv : tuple
        The (mean, log-variance) wrappers of the posterior distribution ``q``.

    Returns
    -------
    torch.Tensor
        The one-sample estimate of the element-wise KL divergence.
    """
    assert isinstance(p_mulv, tuple)
    assert isinstance(q_mulv, tuple)
    p_mu, p_lv = p_mulv
    q_mu, q_lv = q_mulv

    p_std = p_lv.get_std()
    q_std = q_lv.get_std()

    p_distrib = Normal(p_mu.get(), p_std)
    q_distrib = Normal(q_mu.get(), q_std)
    return q_distrib.log_prob(z) - p_distrib.log_prob(z)

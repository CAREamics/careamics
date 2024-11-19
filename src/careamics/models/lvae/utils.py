"""
Script for utility functions needed by the LVAE model.
"""

from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.distributions.normal import Normal


def torch_nanmean(inp):
    return torch.mean(inp[~inp.isnan()])


def power_of_2(self, x):
    assert isinstance(x, int)
    if x == 1:
        return True
    if x == 0:
        # happens with validation
        return False
    if x % 2 == 1:
        return False
    return self.power_of_2(x // 2)


class Enum:
    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def contains(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return True
        return False

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f"{cls.__name__}:{enum_type_str} doesnot exist."


class LossType(Enum):
    Elbo = 0
    ElboWithCritic = 1
    ElboMixedReconstruction = 2
    MSE = 3
    ElboWithNbrConsistency = 4
    ElboSemiSupMixedReconstruction = 5
    ElboCL = 6
    ElboRestrictedReconstruction = 7
    DenoiSplitMuSplit = 8


class ModelType(Enum):
    LadderVae = 3
    LadderVaeTwinDecoder = 4
    LadderVAECritic = 5
    # Separate vampprior: two optimizers
    LadderVaeSepVampprior = 6
    # one encoder for mixed input, two for separate inputs.
    LadderVaeSepEncoder = 7
    LadderVAEMultiTarget = 8
    LadderVaeSepEncoderSingleOptim = 9
    UNet = 10
    BraveNet = 11
    LadderVaeStitch = 12
    LadderVaeSemiSupervised = 13
    LadderVaeStitch2Stage = 14  # Note that previously trained models will have issue.
    # since earlier, LadderVaeStitch2Stage = 13, LadderVaeSemiSupervised = 14
    LadderVaeMixedRecons = 15
    LadderVaeCL = 16
    LadderVaeTwoDataSet = (
        17  # on one subdset, apply disentanglement, on other apply reconstruction
    )
    LadderVaeTwoDatasetMultiBranch = 18
    LadderVaeTwoDatasetMultiOptim = 19
    LVaeDeepEncoderIntensityAug = 20
    AutoRegresiveLadderVAE = 21
    LadderVAEInterleavedOptimization = 22
    Denoiser = 23
    DenoiserSplitter = 24
    SplitterDenoiser = 25
    LadderVAERestrictedReconstruction = 26
    LadderVAETwoDataSetRestRecon = 27
    LadderVAETwoDataSetFinetuning = 28


def _pad_crop_img(
    x: torch.Tensor, size: Sequence[int], mode: Literal["crop", "pad"]
) -> torch.Tensor:
    """Pads or crops a tensor.

    Pads or crops a tensor of shape (B, C, [Z], Y, X) to new shape.

    Parameters:
    -----------
    x: torch.Tensor
        Input image of shape (B, C, [Z], Y, X)
    size: Sequence[int]
        Desired size ([Z*], Y*, X*)
    mode: Literal["crop", "pad"]
        Mode, either 'pad' or 'crop'

    Returns:
    --------
    torch.Tensor:
        The padded or cropped tensor
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

    diffs = [abs(x - s) for x, s in zip(x_size, size)]
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
    """Pads a tensor

    Pads a tensor of shape (B, C, [Z], Y, X) to desired spatial dimensions.

    Parameters:
    -----------
        x (torch.Tensor): Input image of shape (B, C, [Z], Y, X)
        size (list or tuple): Desired size  ([Z*], Y*, X*)

    Returns:
    --------
        The padded tensor
    """
    return _pad_crop_img(x, size, "pad")


def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to a desired height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)

    Returns
    -------
        The cropped tensor
    """
    return _pad_crop_img(x, size, "crop")


class StableExponential:
    """
    Class that redefines the definition of exp() to increase numerical stability.
    Naturally, also the definition of log() must change accordingly.
    However, it is worth noting that the two operations remain one the inverse of the other,
    meaning that x = log(exp(x)) and x = exp(log(x)) are always true.

    Definition:
        exp(x) = {
            exp(x) if x<=0
            x+1    if x>0
        }

        log(x) = {
            x        if x<=0
            log(1+x) if x>0
        }

    NOTE 1:
        Within the class everything is done on the tensor given as input to the constructor.
        Therefore, when exp() is called, self._tensor.exp() is computed.
        When log() is called, torch.log(self._tensor.exp()) is computed instead.

    NOTE 2:
        Given the output from exp(), torch.log() or the log() method of the class give identical results.
    """

    def __init__(self, tensor):
        self._raw_tensor = tensor
        posneg_dic = self.posneg_separation(self._raw_tensor)
        self.pos_f, self.neg_f = posneg_dic["filter"]
        self.pos_data, self.neg_data = posneg_dic["value"]

    def posneg_separation(self, tensor):
        pos = tensor > 0
        pos_tensor = torch.clip(tensor, min=0)

        neg = tensor <= 0
        neg_tensor = torch.clip(tensor, max=0)

        return {"filter": [pos, neg], "value": [pos_tensor, neg_tensor]}

    def exp(self):
        return torch.exp(self.neg_data) * self.neg_f + (1 + self.pos_data) * self.pos_f

    def log(self):
        return self.neg_data * self.neg_f + torch.log(1 + self.pos_data) * self.pos_f


class StableLogVar:
    """
    Class that provides a numerically stable implementation of Log-Variance.
    Specifically, it uses the exp() and log() formulas defined in `StableExponential` class.
    """

    def __init__(
        self, logvar: torch.Tensor, enable_stable: bool = True, var_eps: float = 1e-6
    ):
        """
        Constructor.

        Parameters
        ----------
        logvar: torch.Tensor
            The input (true) logvar vector, to be converted in the Stable version.
        enable_stable: bool, optional
            Whether to compute the stable version of log-variance. Default is `True`.
        var_eps: float, optional
            The minimum value attainable by the variance. Default is `1e-6`.
        """
        self._lv = logvar
        self._enable_stable = enable_stable
        self._eps = var_eps

    def get(self) -> torch.Tensor:
        if self._enable_stable is False:
            return self._lv

        return torch.log(self.get_var())

    def get_var(self) -> torch.Tensor:
        """
        Get Variance from Log-Variance.
        """
        if self._enable_stable is False:
            return torch.exp(self._lv)
        return StableExponential(self._lv).exp() + self._eps

    def get_std(self) -> torch.Tensor:
        return torch.sqrt(self.get_var())

    @property
    def is_3D(self) -> bool:
        """Check if the _lv tensor is 3D.

        Recall that, in this framework, tensors have shape (B, C, [Z], Y, X).
        """
        return self._lv.dim() == 5

    def centercrop_to_size(self, size: Sequence[int]) -> None:
        """
        Centercrop the log-variance tensor to the desired size.

        Parameters
        ----------
        size: torch.Tensor
            The desired size of the log-variance tensor.
        """
        assert not self.is_3D, "Centercrop is implemented only for 2D tensors."

        if self._lv.shape[-1] == size:
            return

        diff = self._lv.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._lv = F.center_crop(self._lv, (size, size))


class StableMean:

    def __init__(self, mean):
        self._mean = mean

    def get(self) -> torch.Tensor:
        return self._mean

    @property
    def is_3D(self) -> bool:
        """Check if the _mean tensor is 3D.

        Recall that, in this framework, tensors have shape (B, C, [Z], Y, X).
        """
        return self._mean.dim() == 5

    def centercrop_to_size(self, size: Sequence[int]) -> None:
        """Centercrop the mean tensor to the desired size.

        Implemented only in the case of 2D tensors.

        Parameters
        ----------
        size: torch.Tensor
            The desired size of the log-variance tensor.
        """
        assert not self.is_3D, "Centercrop is implemented only for 2D tensors."

        if self._mean.shape[-1] == size:
            return

        diff = self._mean.shape[-1] - size
        assert diff > 0 and diff % 2 == 0
        self._mean = F.center_crop(self._mean, (size, size))


def allow_numpy(func):
    """
    All optional arguments are passed as is. positional arguments are checked. if they are numpy array,
    they are converted to torch Tensor.
    """

    def numpy_wrapper(*args, **kwargs):
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
    """Wrapper for torch.nn.functional.interpolate."""

    def __init__(self, size=None, scale=None, mode="bilinear", align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return out


def kl_normal_mc(z, p_mulv, q_mulv):
    """
    One-sample estimation of element-wise KL between two diagonal
    multivariate normal distributions. Any number of dimensions,
    broadcasting supported (be careful).
    :param z:
    :param p_mulv:
    :param q_mulv:
    :return:
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

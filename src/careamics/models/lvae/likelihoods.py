"""
Script containing modules for definining different likelihood functions (as nn.Module).
"""

import math
from typing import Dict, Literal, Tuple, Union

import numpy as np
import torch
from torch import nn


class LikelihoodModule(nn.Module):
    """
    The base class for all likelihood modules.
    It defines the fundamental structure and methods for specialized likelihood models.
    """

    def distr_params(self, x):
        return None

    def set_params_to_same_device_as(self, correct_device_tensor):
        pass

    @staticmethod
    def logvar(params):
        return None

    @staticmethod
    def mean(params):
        return None

    @staticmethod
    def mode(params):
        return None

    @staticmethod
    def sample(params):
        return None

    def log_likelihood(self, x, params):
        return None

    def forward(
        self, input_: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        distr_params = self.distr_params(input_)
        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        logvar = self.logvar(distr_params)

        if x is None:
            ll = None
        else:
            ll = self.log_likelihood(x, distr_params)

        dct = {
            "mean": mean,
            "mode": mode,
            "sample": sample,
            "params": distr_params,
            "logvar": logvar,
        }

        return ll, dct


class GaussianLikelihood(LikelihoodModule):
    r"""
    A specialize `LikelihoodModule` for Gaussian likelihood.

    Specifically, in the LVAE model, the likelihood is defined as:
        p(x|z_1) = N(x|\mu_{p,1}, \sigma_{p,1}^2)
    """

    def __init__(
        self,
        ch_in: int,
        color_channels: int,
        predict_logvar: Literal[None, "pixelwise", "global", "channelwise"] = None,
        logvar_lowerbound: float = None,
        conv2d_bias: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        predict_logvar: Literal[None, 'global', 'pixelwise', 'channelwise'], optional
            If not `None`, it expresses how to compute the log-variance.
            Namely:
            - if `pixelwise`, log-variance is computed for each pixel.
            - if `global`, log-variance is computed as the mean of all pixel-wise entries.
            - if `channelwise`, log-variance is computed as the average over the channels.
            Default is `None`.
        logvar_lowerbound: float, optional
            The lowerbound value for log-variance. Default is `None`.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        """
        super().__init__()

        # If True, then we also predict pixelwise logvar.
        self.predict_logvar = predict_logvar
        self.logvar_lowerbound = logvar_lowerbound
        self.conv2d_bias = conv2d_bias
        assert self.predict_logvar in [None, "global", "pixelwise", "channelwise"]

        # logvar_ch_needed = self.predict_logvar is not None
        # self.parameter_net = nn.Conv2d(ch_in,
        #                                color_channels * (1 + logvar_ch_needed),
        #                                kernel_size=3,
        #                                padding=1,
        #                                bias=self.conv2d_bias)
        self.parameter_net = nn.Identity()

        print(
            f"[{self.__class__.__name__}] PredLVar:{self.predict_logvar} LowBLVar:{self.logvar_lowerbound}"
        )

    def get_mean_lv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the output of the top-down pass, compute the mean and log-variance of the
        Gaussian distribution defining the likelihood.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor to the likelihood module, i.e., the output of the top-down pass.
        """
        # Feed the output of the top-down pass to a parameter network
        # This network can be either a Conv2d or Identity module
        x = self.parameter_net(x)

        if self.predict_logvar is not None:
            # Get pixel-wise mean and logvar
            mean, lv = x.chunk(2, dim=1)

            # Optionally, compute the global or channel-wise logvar
            if self.predict_logvar in ["channelwise", "global"]:
                if self.predict_logvar == "channelwise":
                    # logvar should be of the following shape (batch, num_channels, ). Other dims would be singletons.
                    N = np.prod(lv.shape[:2])
                    new_shape = (*mean.shape[:2], *([1] * len(mean.shape[2:])))
                elif self.predict_logvar == "global":
                    # logvar should be of the following shape (batch, ). Other dims would be singletons.
                    N = lv.shape[0]
                    new_shape = (*mean.shape[:1], *([1] * len(mean.shape[1:])))
                else:
                    raise ValueError(
                        f"Invalid value for self.predict_logvar:{self.predict_logvar}"
                    )

                lv = torch.mean(lv.reshape(N, -1), dim=1)
                lv = lv.reshape(new_shape)

            # Optionally, clip log-var to a lower bound
            if self.logvar_lowerbound is not None:
                lv = torch.clip(lv, min=self.logvar_lowerbound)
        else:
            mean = x
            lv = None
        return mean, lv

    def distr_params(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get parameters (mean, log-var) of the Gaussian distribution defined by the likelihood.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor to the likelihood module, i.e., the output of the top-down pass.
        """
        mean, lv = self.get_mean_lv(x)
        params = {
            "mean": mean,
            "logvar": lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params["mean"]

    @staticmethod
    def mode(params):
        return params["mean"]

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params["mean"]

    @staticmethod
    def logvar(params):
        return params["logvar"]

    def log_likelihood(self, x, params):
        if self.predict_logvar is not None:
            logprob = log_normal(x, params["mean"], params["logvar"])
        else:
            logprob = -0.5 * (params["mean"] - x) ** 2
        return logprob


def log_normal(
    x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log-probability at `x` of a Gaussian distribution
    with parameters `(mean, exp(logvar))`.

    NOTE: In the case of LVAE, the log-likeihood formula becomes:
        \\mathbb{E}_{z_1\\sim{q_\\phi}}[\\log{p_\theta(x|z_1)}]=-\frac{1}{2}(\\mathbb{E}_{z_1\\sim{q_\\phi}}[\\log{2\\pi\\sigma_{p,0}^2(z_1)}] +\\mathbb{E}_{z_1\\sim{q_\\phi}}[\frac{(x-\\mu_{p,0}(z_1))^2}{\\sigma_{p,0}^2(z_1)}])

    Parameters
    ----------
    x: torch.Tensor
        The ground-truth tensor. Shape is (batch, channels, dim1, dim2).
    mean: torch.Tensor
        The inferred mean of distribution. Shape is (batch, channels, dim1, dim2).
    logvar: torch.Tensor
        The inferred log-variance of distribution. Shape has to be either scalar or broadcastable.
    """
    var = torch.exp(logvar)
    log_prob = -0.5 * (
        ((x - mean) ** 2) / var + logvar + torch.tensor(2 * math.pi).log()
    )
    return log_prob


class NoiseModelLikelihood(LikelihoodModule):

    def __init__(
        self,
        ch_in: int,
        color_channels: int,
        data_mean: Union[Dict[str, torch.Tensor], torch.Tensor],
        data_std: Union[Dict[str, torch.Tensor], torch.Tensor],
        noiseModel: nn.Module,
    ):
        super().__init__()
        self.parameter_net = (
            nn.Identity()
        )  # nn.Conv2d(ch_in, color_channels, kernel_size=3, padding=1)
        self.data_mean = data_mean
        self.data_std = data_std
        self.noiseModel = noiseModel

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)
        elif isinstance(self.data_mean, dict):
            for key in self.data_mean.keys():
                self.data_mean[key] = self.data_mean[key].to(
                    correct_device_tensor.device
                )
                self.data_std[key] = self.data_std[key].to(correct_device_tensor.device)

    def get_mean_lv(self, x):
        return self.parameter_net(x), None

    def distr_params(self, x):
        mean, lv = self.get_mean_lv(x)
        # mean, lv = x.chunk(2, dim=1)

        params = {
            "mean": mean,
            "logvar": lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params["mean"]

    @staticmethod
    def mode(params):
        return params["mean"]

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params["mean"]

    def log_likelihood(self, x: torch.Tensor, params: Dict[str, torch.Tensor]):
        """
        Compute the log-likelihood given the parameters `params` obtained from the reconstruction tensor and the target tensor `x`.
        """
        predicted_s_denormalized = (
            params["mean"] * self.data_std["target"] + self.data_mean["target"]
        )
        x_denormalized = x * self.data_std["target"] + self.data_mean["target"]
        # predicted_s_cloned = predicted_s_denormalized
        # predicted_s_reduced = predicted_s_cloned.permute(1, 0, 2, 3)

        # x_cloned = x_denormalized
        # x_cloned = x_cloned.permute(1, 0, 2, 3)
        # x_reduced = x_cloned[0, ...]
        # import pdb;pdb.set_trace()
        likelihoods = self.noiseModel.likelihood(
            x_denormalized, predicted_s_denormalized
        )
        # likelihoods = self.noiseModel.likelihood(x, params['mean'])
        logprob = torch.log(likelihoods)
        return logprob

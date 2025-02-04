"""Script containing the common basic blocks (nn.Module)
reused by the LadderVAE architecture.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from .utils import (
    StableLogVar,
    StableMean,
    kl_normal_mc,
)

ConvType = Union[nn.Conv2d, nn.Conv3d]
NormType = Union[nn.BatchNorm2d, nn.BatchNorm3d]
DropoutType = Union[nn.Dropout2d, nn.Dropout3d]


class NormalStochasticBlock(nn.Module):
    """
    Stochastic block used in the Top-Down inference pass.

    Algorithm:
        - map input parameters to q(z) and (optionally) p(z) via convolution
        - sample a latent tensor z ~ q(z)
        - feed z to convolution and return.

    NOTE 1:
        If parameters for q are not given, sampling is done from p(z).

    NOTE 2:
        The restricted KL divergence is obtained by first computing the element-wise KL divergence
        (i.e., the KL computed for each element of the latent tensors). Then, the restricted version
        is computed by summing over the channels and the spatial dimensions associated only to the
        portion of the latent tensor that is used for prediction.
    """

    def __init__(
        self,
        c_in: int,
        c_vars: int,
        c_out: int,
        conv_dims: int = 2,
        kernel: int = 3,
        transform_p_params: bool = True,
        vanilla_latent_hw: int = None,
        use_naive_exponential: bool = False,
    ):
        """
        Parameters
        ----------
        c_in: int
            The number of channels of the input tensor.
        c_vars: int
            The number of channels of the latent space tensor.
        c_out:  int
            The output of the stochastic layer.
            Note that this is different from the sampled latent z.
        conv_dims: int, optional
            The number of dimensions of the convolutional layers (2D or 3D).
            Default is 2.
        kernel: int, optional
            The size of the kernel used in convolutional layers.
            Default is 3.
        transform_p_params: bool, optional
            Whether a transformation should be applied to the `p_params` tensor.
            The transformation consists in a 2D convolution ()`conv_in_p()`) that
            maps the input to a larger number of channels.
            Default is `True`.
        vanilla_latent_hw: int, optional
            The shape of the latent tensor used for prediction (i.e., it influences the computation of restricted KL).
            Default is `None`.
        use_naive_exponential: bool, optional
            If `False`, exponentials are computed according to the alternative definition
            provided by `StableExponential` class. This should improve numerical stability
            in the training process. Default is `False`.
        """
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self.conv_dims = conv_dims
        self._use_naive_exponential = use_naive_exponential
        self._vanilla_latent_hw = vanilla_latent_hw

        conv_layer: ConvType = getattr(nn, f"Conv{conv_dims}d")

        if transform_p_params:
            self.conv_in_p = conv_layer(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = conv_layer(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = conv_layer(c_vars, c_out, kernel, padding=pad)

    def get_z(
        self,
        sampling_distrib: torch.distributions.normal.Normal,
        forced_latent: Union[torch.Tensor, None],
        mode_pred: bool,
        use_uncond_mode: bool,
    ) -> torch.Tensor:
        """Sample a latent tensor from the given latent distribution.

        Latent tensor can be obtained is several ways:
            - Sampled from the (Gaussian) latent distribution.
            - Taken as a pre-defined forced latent.
            - Taken as the mode (mean) of the latent distribution.
            - In prediction mode (`mode_pred==True`), can be either sample or taken as the distribution mode.

        Parameters
        ----------
        sampling_distrib: torch.distributions.normal.Normal
            The Gaussian distribution from which latent tensor is sampled.
        forced_latent: torch.Tensor
            A pre-defined latent tensor. If it is not `None`, than it is used as the actual latent tensor and,
            hence, sampling does not happen.
        mode_pred: bool
            Whether the model is prediction mode.
        use_uncond_mode: bool
            Whether to use the uncoditional distribution p(z) to sample latents in prediction mode.
        """
        if forced_latent is None:
            if mode_pred:
                if use_uncond_mode:
                    z = sampling_distrib.mean
                else:
                    z = sampling_distrib.rsample()
            else:
                z = sampling_distrib.rsample()
        else:
            z = forced_latent
        return z

    def sample_from_q(
        self, q_params: torch.Tensor, var_clip_max: float
    ) -> torch.Tensor:
        """
        Given an input parameter tensor defining q(z),
        it processes it by calling `process_q_params()` method and
        sample a latent tensor from the resulting distribution.

        Parameters
        ----------
        q_params: torch.Tensor
            The input tensor to be processed.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        _, _, q = self.process_q_params(q_params, var_clip_max)
        return q.rsample()

    def compute_kl_metrics(
        self,
        p: torch.distributions.normal.Normal,
        p_params: torch.Tensor,
        q: torch.distributions.normal.Normal,
        q_params: torch.Tensor,
        mode_pred: bool,
        analytical_kl: bool,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute KL (analytical or MC estimate) and then process it, extracting composed versions of the metric.
        Specifically, the different versions of the KL loss terms are:
            - `kl_elementwise`: KL term for each single element of the latent tensor [Shape: (batch, ch, h, w)].
            - `kl_samplewise`: KL term associated to each sample in the batch [Shape: (batch, )].
            - `kl_samplewise_restricted`: KL term only associated to the portion of the latent tensor that is
            used for prediction and summed over channel and spatial dimensions [Shape: (batch, )].
            - `kl_channelwise`: KL term associated to each sample and each channel [Shape: (batch, ch, )].
            - `kl_spatial`: KL term summed over the channels, i.e., retaining the spatial dimensions [Shape: (batch, h, w)]

        Parameters
        ----------
        p: torch.distributions.normal.Normal
            The prior generative distribution p(z_i|z_{i+1}) (or p(z_L)).
        p_params: torch.Tensor
            The parameters of the prior generative distribution.
        q: torch.distributions.normal.Normal
            The inference distribution q(z_i|z_{i+1}) (or q(z_L|x)).
        q_params: torch.Tensor
            The parameters of the inference distribution.
        mode_pred: bool
            Whether the model is in prediction mode.
        analytical_kl: bool
            Whether to compute the KL divergence analytically or using Monte Carlo estimation.
        z: torch.Tensor
            The sampled latent tensor.
        """
        kl_samplewise_restricted = None
        if mode_pred is False:  # if not predicting
            if analytical_kl:
                kl_elementwise = kl_divergence(q, p)
            else:
                kl_elementwise = kl_normal_mc(z, p_params, q_params)

            all_dims = tuple(range(len(kl_elementwise.shape)))
            kl_samplewise = kl_elementwise.sum(all_dims[1:])
            kl_channelwise = kl_elementwise.sum(all_dims[2:])

            # compute KL only on the portion of the latent space that is used for prediction.
            pad = (kl_elementwise.shape[-1] - self._vanilla_latent_hw) // 2
            if pad > 0:
                tmp = kl_elementwise[..., pad:-pad, pad:-pad]
                kl_samplewise_restricted = tmp.sum(all_dims[1:])
            else:
                kl_samplewise_restricted = kl_samplewise

            # Compute spatial KL analytically (but conditioned on samples from
            # previous layers)
            kl_spatial = kl_elementwise.sum(1)
        else:  # if predicting, no need to compute KL
            kl_elementwise = kl_samplewise = kl_spatial = kl_channelwise = None

        kl_dict = {
            "kl_elementwise": kl_elementwise,  # (batch, ch, h, w)
            "kl_samplewise": kl_samplewise,  # (batch, )
            "kl_samplewise_restricted": kl_samplewise_restricted,  # (batch, )
            "kl_spatial": kl_spatial,  # (batch, h, w)
            "kl_channelwise": kl_channelwise,  # (batch, ch)
        }  # TODO revisit, check dims
        return kl_dict

    def process_p_params(
        self, p_params: torch.Tensor, var_clip_max: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.normal.Normal]:
        """Process the input parameters to get the prior distribution p(z_i|z_{i+1}) (or p(z_L)).

        Processing consists in:
            - (optionally) 2D convolution on the input tensor to increase number of channels.
            - split the resulting tensor into two chunks, the mean and the log-variance.
            - (optionally) clip the log-variance to an upper threshold.
            - define the normal distribution p(z) given the parameter tensors above.

        Parameters
        ----------
        p_params: torch.Tensor
            The input tensor to be processed.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        if var_clip_max is not None:
            p_lv = torch.clip(p_lv, max=var_clip_max)

        p_mu = StableMean(p_mu)
        p_lv = StableLogVar(p_lv, enable_stable=not self._use_naive_exponential)
        p = Normal(p_mu.get(), p_lv.get_std())
        return p_mu, p_lv, p

    def process_q_params(
        self, q_params: torch.Tensor, var_clip_max: float, allow_oddsizes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.normal.Normal]:
        """
        Process the input parameters to get the inference distribution q(z_i|z_{i+1}) (or q(z|x)).

        Processing consists in:
            - convolution on the input tensor to double the number of channels.
            - split the resulting tensor into 2 chunks, respectively mean and log-var.
            - (optionally) clip the log-variance to an upper threshold.
            - (optionally) crop the resulting tensors to ensure that the last spatial dimension is even.
            - define the normal distribution q(z) given the parameter tensors above.

        Parameters
        ----------
        p_params: torch.Tensor
            The input tensor to be processed.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        q_params = self.conv_in_q(q_params)

        q_mu, q_lv = q_params.chunk(2, dim=1)
        if var_clip_max is not None:
            q_lv = torch.clip(q_lv, max=var_clip_max)

        if q_mu.shape[-1] % 2 == 1 and allow_oddsizes is False:
            q_mu = F.center_crop(q_mu, q_mu.shape[-1] - 1)
            q_lv = F.center_crop(q_lv, q_lv.shape[-1] - 1)
            # TODO revisit ?!
        q_mu = StableMean(q_mu)
        q_lv = StableLogVar(q_lv, enable_stable=not self._use_naive_exponential)
        q = Normal(q_mu.get(), q_lv.get_std())
        return q_mu, q_lv, q

    def forward(
        self,
        p_params: torch.Tensor,
        q_params: Union[torch.Tensor, None] = None,
        forced_latent: Union[torch.Tensor, None] = None,
        force_constant_output: bool = False,
        analytical_kl: bool = False,
        mode_pred: bool = False,
        use_uncond_mode: bool = False,
        var_clip_max: Union[float, None] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        p_params: torch.Tensor
            The output tensor of the top-down layer above (i.e., mu_{p,i+1}, sigma_{p,i+1}).
        q_params: torch.Tensor, optional
            The tensor resulting from merging the bu_value tensor at the same hierarchical level
            from the bottom-up pass and the `p_params` tensor. Default is `None`.
        forced_latent: torch.Tensor, optional
            A pre-defined latent tensor. If it is not `None`, than it is used as the actual latent
            tensor and, hence, sampling does not happen. Default is `None`.
        force_constant_output: bool, optional
            Whether to copy the first sample (and rel. distrib parameters) over the whole batch.
            This is used when doing experiment from the prior - q is not used.
            Default is `False`.
        analytical_kl: bool, optional
            Whether to compute the KL divergence analytically or using Monte Carlo estimation.
            Default is `False`.
        mode_pred: bool, optional
            Whether the model is in prediction mode. Default is `False`.
        use_uncond_mode: bool, optional
            Whether to use the uncoditional distribution p(z) to sample latents in prediction mode.
            Default is `False`.
        var_clip_max: float, optional
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped. Default is `None`.
        """
        debug_qvar_max = 0

        # Check sampling options consistency
        assert forced_latent is None

        # Get generative distribution p(z_i|z_{i+1})
        p_mu, p_lv, p = self.process_p_params(p_params, var_clip_max)
        p_params = (p_mu, p_lv)

        if q_params is not None:
            # Get inference distribution q(z_i|z_{i+1})
            q_mu, q_lv, q = self.process_q_params(q_params, var_clip_max)
            q_params = (q_mu, q_lv)
            debug_qvar_max = torch.max(q_lv.get())
            sampling_distrib = q
            q_size = q_mu.get().shape[-1]
            if p_mu.get().shape[-1] != q_size and mode_pred is False:
                p_mu.centercrop_to_size(q_size)
                p_lv.centercrop_to_size(q_size)
        else:
            sampling_distrib = p

        # Sample latent variable
        z = self.get_z(sampling_distrib, forced_latent, mode_pred, use_uncond_mode)

        # TODO: not necessary, remove
        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = (
                p_params[0][0:1].expand_as(p_params[0]).clone(),
                p_params[1][0:1].expand_as(p_params[1]).clone(),
            )

        # Pass the sampled latent through the output convolution of stochastic block
        out = self.conv_out(z)

        if q_params is not None:
            # Compute log q(z)
            logprob_q = q.log_prob(z).sum(tuple(range(1, z.dim())))
            # Compute KL divergence metrics
            kl_dict = self.compute_kl_metrics(
                p, p_params, q, q_params, mode_pred, analytical_kl, z
            )
        else:
            kl_dict = {}
            logprob_q = None

        # Store meaningful quantities for later computation
        data = kl_dict
        data["z"] = z  # sampled variable at this layer (B, C, [Z], Y, X)
        data["p_params"] = p_params  # (B, C, [Z], Y, X) where B is 1 or batch size
        data["q_params"] = q_params  # (B, C, [Z], Y, X)
        data["logprob_q"] = logprob_q  # (B, )
        data["qvar_max"] = debug_qvar_max
        return out, data

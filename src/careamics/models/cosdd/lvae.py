"""COSDD Ladder VAE.

Ported from the COSDD reference implementation
(https://github.com/krulllab/COSDD, `models/lvae.py`), MIT License,
Copyright (c) 2023 krulllab. Derived from code under the following license:

MIT License

Copyright (c) 2019 Andrea Dittadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .layers import (
    BottomUpLayer,
    Conv,
    DistributionParams,
    ResBlockWithResampling,
    VAETopDownLayer,
)
from .utils import get_padded_size, spatial_pad_crop


class LadderVAE(nn.Module):
    """Ladder Variational Autoencoder (LVAE) encoding the signal into a latent code.

    Parameters
    ----------
    colour_channels : int
        Number of input image channels.
    img_size : Sequence of int
        Spatial size of the training images, used to size the top prior.
    s_code_channels : int
        Number of channels in the returned latent code.
    downsampling : Sequence of Sequence of int
        Binary downsampling per layer and spatial axis, of shape
        (n_layers, dimensions).
    z_dims : Sequence of int or None, default=None
        Dimensions of the latent variables z of each layer. If `None`, 12 layers of
        32 channels are used.
    blocks_per_layer : int, default=1
        Number of residual blocks per layer.
    n_filters : int, default=64
        Number of filters in the convolutional layers.
    scale_initialisation : bool, default=False
        For stability, scale the output of the residual blocks by
        1/(channels**0.5). See VDVAE, Child 2020.
    learn_top_prior : bool, default=True
        Whether to learn the top prior.
    stochastic_skip : bool, default=True
        Whether to use stochastic skip connections.
    checkpointed : bool, default=False
        Whether to use activation checkpointing in the forward pass.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        colour_channels: int,
        img_size: Sequence[int],
        s_code_channels: int,
        downsampling: Sequence[Sequence[int]],
        z_dims: Sequence[int] | None = None,
        blocks_per_layer: int = 1,
        n_filters: int = 64,
        scale_initialisation: bool = False,
        learn_top_prior: bool = True,
        stochastic_skip: bool = True,
        checkpointed: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        colour_channels : int
            Number of input image channels.
        img_size : Sequence of int
            Spatial size of the training images, used to size the top prior.
        s_code_channels : int
            Number of channels in the returned latent code.
        downsampling : Sequence of Sequence of int
            Binary downsampling per layer and spatial axis, of shape
            (n_layers, dimensions).
        z_dims : Sequence of int or None, default=None
            Dimensions of the latent variables z of each layer. If `None`, 12 layers of
            32 channels are used.
        blocks_per_layer : int, default=1
            Number of residual blocks per layer.
        n_filters : int, default=64
            Number of filters in the convolutional layers.
        scale_initialisation : bool, default=False
            For stability, scale the output of the residual blocks by
            1/(channels**0.5). See VDVAE, Child 2020.
        learn_top_prior : bool, default=True
            Whether to learn the top prior.
        stochastic_skip : bool, default=True
            Whether to use stochastic skip connections.
        checkpointed : bool, default=False
            Whether to use activation checkpointing in the forward pass.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        if z_dims is None:
            z_dims = [32] * 12
        super().__init__()
        self.img_size = tuple(img_size)
        self.z_dims = list(z_dims)
        self.n_layers = len(self.z_dims)
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.stochastic_skip = stochastic_skip
        self.checkpointed = checkpointed

        # Number of downsampling steps per layer
        n_dim = len(downsampling[0])
        self.n_downsample = [0 for _ in range(n_dim)]
        for i in range(n_dim):
            for j in range(len(downsampling)):
                self.n_downsample[i] += downsampling[j][i]

        assert len(downsampling) == self.n_layers

        self.first_bottom_up = nn.Sequential(
            Conv(
                colour_channels,
                n_filters,
                5,
                padding=2,
                padding_mode="replicate",
                dimensions=dimensions,
            ),
            nn.Mish(),
            ResBlockWithResampling(
                c_in=n_filters,
                c_out=n_filters,
                gated=False,
                n_resample=[0] * dimensions,
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            ),
        )

        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        for i in range(self.n_layers):
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

            # Add top-down stochastic layer at level i.
            # The architecture when doing inference is roughly as follows:
            #    p_params = output of top-down layer above
            #    bu = inferred bottom-up value at this layer
            #    q_params = merge(bu, p_params)
            #    z = stochastic_layer(q_params)
            #    possibly get skip connection from previous top-down layer
            #    top-down deterministic ResNet
            #
            # When doing generation only, the value bu is not available, the
            # merge layer is not used, and z is sampled directly from p_params.
            self.top_down_layers.append(
                VAETopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    upsampling_steps=downsampling[i],
                    stochastic_skip=stochastic_skip,
                    learn_top_prior=learn_top_prior,
                    top_prior_param_size=self.get_top_prior_param_size(),
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

        self.final_top_down = Conv(
            in_channels=n_filters,
            out_channels=s_code_channels,
            kernel_size=1,
            dimensions=dimensions,
        )

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Encode the input into a latent signal code.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (N, C, *spatial).

        Returns
        -------
        dict of {str: Any}
            Dictionary with the approximate posteriors (`q_list`), the priors
            (`p_list`) and the latent signal code (`s_code`).
        """
        # Pad x to have base 2 side lengths to make resampling steps simpler
        # Save size to crop back down later
        x_size = x.size()[2:]
        padded_size = get_padded_size(x_size, self.n_downsample)
        x = spatial_pad_crop(x, padded_size)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x)

        z, q_list, p_list = self.topdown_pass(bu_values)

        # Restore original image size
        z = spatial_pad_crop(z, x_size)

        output = {
            "q_list": q_list,
            "p_list": p_list,
            "s_code": z,
        }
        return output

    def bottomup_pass(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Compute the bottom-up values of each layer.

        Parameters
        ----------
        x : torch.Tensor
            Padded input tensor.

        Returns
        -------
        list of torch.Tensor
            Bottom-up values, from the bottom to the top layer.
        """
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            if i % 2 == 0 and self.checkpointed:
                x = checkpoint(  # type: ignore[assignment]
                    self.bottom_up_layers[i],
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(
        self,
        bu_values: Sequence[torch.Tensor | None] | None = None,
        n_img_prior: int | None = None,
    ) -> tuple[torch.Tensor, list[DistributionParams], list[DistributionParams]]:
        """Compute the top-down pass, sampling the latent variables.

        Parameters
        ----------
        bu_values : Sequence of torch.Tensor or None, default=None
            Bottom-up values of each layer. If `None`, samples are drawn from the
            prior.
        n_img_prior : int or None, default=None
            Number of images to generate when sampling from the prior.

        Returns
        -------
        torch.Tensor
            Latent signal code.
        list of dict
            Approximate posterior distributions, from the top to the bottom layer.
        list of dict
            Prior distributions, from the top to the bottom layer.
        """
        if bu_values is None:
            bu_values = [None] * self.n_layers

        # For the KL divergence of each layer
        q_list = []
        p_list = []

        p_params: torch.Tensor | None = None
        for i in reversed(range(self.n_layers)):
            skip_input = p_params

            if i % 2 == 0 and self.checkpointed:
                z, q, p = checkpoint(  # type: ignore[misc]
                    self.top_down_layers[i],
                    p_params,
                    skip_input,
                    bu_values[i],
                    n_img_prior,
                    use_reentrant=False,
                )
            else:
                z, q, p = self.top_down_layers[i](
                    p_params,
                    skip_input,
                    bu_values[i],
                    n_img_prior,
                )
            p_params = z
            q_list.append(q)
            p_list.append(p)

        z = self.final_top_down(z)

        return z, q_list, p_list

    def get_top_prior_param_size(self) -> list[int]:
        """Return the size of the top prior parameters.

        Returns
        -------
        list of int
            Size of the top prior parameters, (1, 2 * z_dim, *spatial).
        """
        padded_size = get_padded_size(self.img_size, self.n_downsample)
        dwnsc = [2**nd for nd in self.n_downsample]
        top_prior_size = [s // d for s, d in zip(padded_size, dwnsc, strict=True)]
        c = self.z_dims[-1] * 2  # mu and log-sigma
        top_prior_size = [1, c, *top_prior_size]

        return top_prior_size

    @torch.no_grad()
    def sample_from_prior(self, n_images: int) -> torch.Tensor:
        """Sample latent signal codes from the prior.

        Parameters
        ----------
        n_images : int
            Number of codes to sample.

        Returns
        -------
        torch.Tensor
            Sampled latent signal codes, cropped to `img_size`.
        """
        z, _, _ = self.topdown_pass(n_img_prior=n_images)
        generated_s_code = spatial_pad_crop(z, self.img_size)

        return generated_s_code

    def kl_divergence(
        self,
        q_list: Sequence[DistributionParams],
        p_list: Sequence[DistributionParams],
    ) -> torch.Tensor:
        """Sum of the KL divergences between the posteriors and priors of all layers.

        Parameters
        ----------
        q_list : Sequence of dict
            Approximate posterior distributions.
        p_list : Sequence of dict
            Prior distributions.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence.
        """
        kl_terms = [
            kl_divergence(q, p).sum() for q, p in zip(q_list, p_list, strict=True)
        ]
        kl_sum = kl_terms[0]
        for kl_term in kl_terms[1:]:
            kl_sum = kl_sum + kl_term
        return kl_sum


def kl_divergence(q: DistributionParams, p: DistributionParams) -> torch.Tensor:
    """Element-wise KL divergence between two normal distributions.

    Parameters
    ----------
    q : dict of {str: torch.Tensor}
        Mean (`mu`) and standard deviation (`std`) of the first distribution.
    p : dict of {str: torch.Tensor}
        Mean (`mu`) and standard deviation (`std`) of the second distribution.

    Returns
    -------
    torch.Tensor
        Element-wise KL(q || p).
    """
    q_std, q_mu, p_std, p_mu = q["std"], q["mu"], p["std"], p["mu"]
    assert q_std is not None and q_mu is not None
    assert p_std is not None and p_mu is not None
    q_logstd = torch.log(q_std)
    p_logstd = torch.log(p_std)
    var_ratio = torch.exp(q_logstd - p_logstd).pow(2)
    logstd_diff = q_logstd - p_logstd
    t1 = (q_mu - p_mu).pow(2) / p_std.pow(2)
    return 0.5 * (var_ratio + t1 - 1) - logstd_diff

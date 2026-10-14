"""COSDD building blocks.

Ported from the COSDD reference implementation
(https://github.com/krulllab/COSDD, `models/nn.py`), MIT License,
Copyright (c) 2023 krulllab. Parts of the Ladder VAE layers are derived from code
under the following license:

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

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .utils import LinearUpsample, Rotate90, interleave

DistributionParams = dict[str, torch.Tensor | None]
"""Mean (`mu`) and standard deviation (`std`) of a normal distribution."""


class Conv(nn.Module):
    """Convolutional layer for 1, 2, or 3D data.

    Same arguments as `nn.ConvXd`, but the dimensionality of the data is passed as an
    argument.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or Sequence of int
        Size of the convolutional kernel.
    stride : int or Sequence of int, default=1
        Stride of the convolution.
    padding : int or Sequence of int, default=0
        Padding added to the input.
    dilation : int or Sequence of int, default=1
        Spacing between kernel elements.
    groups : int, default=1
        Number of blocked connections from input to output channels.
    bias : bool, default=True
        Whether to add a learnable bias.
    padding_mode : str, default="zeros"
        Padding mode.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or Sequence of int
            Size of the convolutional kernel.
        stride : int or Sequence of int, default=1
            Stride of the convolution.
        padding : int or Sequence of int, default=0
            Padding added to the input.
        dilation : int or Sequence of int, default=1
            Spacing between kernel elements.
        groups : int, default=1
            Number of blocked connections from input to output channels.
        bias : bool, default=True
            Whether to add a learnable bias.
        padding_mode : str, default="zeros"
            Padding mode.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        self.args: dict[str, Any] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
        }
        conv = getattr(nn, f"Conv{dimensions}d")
        self.conv: nn.Module = conv(**self.args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.conv(x)


class GateLayer(nn.Module):
    """Gated activation layer.

    Parameters
    ----------
    channels : int
        Number of channels.
    kernel_size : int
        Size of the convolutional kernel, must be odd.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(self, channels: int, kernel_size: int, dimensions: int = 2) -> None:
        """Constructor.

        Parameters
        ----------
        channels : int
            Number of channels.
        kernel_size : int
            Size of the convolutional kernel, must be odd.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = Conv(
            channels, 2 * channels, kernel_size, padding=pad, dimensions=dimensions
        )
        self.nonlin = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.conv(x)
        x, gate = torch.chunk(x, dim=1, chunks=2)
        x = self.nonlin(x)
        gate = torch.sigmoid(gate)
        return x * gate


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and Mish activations.

    Parameters
    ----------
    channels : int
        Number of channels.
    kernel_size : int, default=3
        Size of the convolutional kernel, must be odd.
    groups : int, default=1
        Number of groups for grouped convolution.
    gated : bool, default=True
        Whether to use a gated activation at the end of the block.
    scale_initialisation : bool, default=False
        For stability, scale the output of the residual block by
        1/(channels**0.5). See VDVAE, Child 2020.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        groups: int = 1,
        gated: bool = True,
        scale_initialisation: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        channels : int
            Number of channels.
        kernel_size : int, default=3
            Size of the convolutional kernel, must be odd.
        groups : int, default=1
            Number of groups for grouped convolution.
        gated : bool, default=True
            Whether to use a gated activation at the end of the block.
        scale_initialisation : bool, default=False
            For stability, scale the output of the residual block by
            1/(channels**0.5). See VDVAE, Child 2020.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        self.channels = channels
        assert kernel_size % 2 == 1
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size
        self.groups = groups
        self.gated = gated
        if scale_initialisation:
            self.rescale = channels**0.5
        else:
            self.rescale = 1.0

        BatchNorm = getattr(nn, f"BatchNorm{dimensions}d")

        self.block = nn.Sequential()
        for _ in range(2):
            self.block.append(BatchNorm(self.channels))
            self.block.append(nn.Mish())
            conv = Conv(
                self.channels,
                self.channels,
                self.kernel_size,
                padding=self.pad,
                groups=self.groups,
                dimensions=dimensions,
            )
            self.block.append(conv)
        if self.gated:
            self.block.append(GateLayer(self.channels, 1, dimensions=dimensions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        out = self.block(x) + x
        out = out / self.rescale
        return out


class ResBlockWithResampling(nn.Module):
    """Residual block with resampling.

    Parameters
    ----------
    c_in : int
        Number of input channels.
    c_out : int
        Number of output channels.
    n_resample : Sequence of int
        Number of resampling steps along each spatial axis.
    resample : {"up", "down"} or None, default=None
        Resampling method.
    res_block_kernel : int, default=3
        Kernel size for the residual block.
    groups : int, default=1
        Number of groups for grouped convolution.
    gated : bool, default=True
        Whether to use gated activation.
    scale_initialisation : bool, default=False
        For stability, scale the output of the residual block by
        1/(channels**0.5). See VDVAE, Child 2020.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        n_resample: Sequence[int],
        resample: Literal["up", "down"] | None = None,
        res_block_kernel: int = 3,
        groups: int = 1,
        gated: bool = True,
        scale_initialisation: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        c_in : int
            Number of input channels.
        c_out : int
            Number of output channels.
        n_resample : Sequence of int
            Number of resampling steps along each spatial axis.
        resample : {"up", "down"} or None, default=None
            Resampling method.
        res_block_kernel : int, default=3
            Kernel size for the residual block.
        groups : int, default=1
            Number of groups for grouped convolution.
        gated : bool, default=True
            Whether to use gated activation.
        scale_initialisation : bool, default=False
            For stability, scale the output of the residual block by
            1/(channels**0.5). See VDVAE, Child 2020.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        assert resample in ["up", "down", None]
        scale_factor = [2**n for n in n_resample]

        self.pre_conv: nn.Module
        if resample == "up":
            self.pre_conv = nn.Sequential(
                LinearUpsample(scale_factor=scale_factor),
                Conv(c_in, c_out, 1, groups=groups, dimensions=dimensions),
            )
        elif resample == "down":
            self.pre_conv = Conv(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=scale_factor,
                groups=groups,
                dimensions=dimensions,
            )
        elif c_in != c_out:
            self.pre_conv = Conv(c_in, c_out, 1, groups=groups, dimensions=dimensions)
        else:
            self.pre_conv = nn.Identity()

        self.res_block = ResidualBlock(
            channels=c_out,
            kernel_size=res_block_kernel,
            groups=groups,
            gated=gated,
            scale_initialisation=scale_initialisation,
            dimensions=dimensions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply resampling followed by the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.pre_conv(x)
        x = self.res_block(x)
        return x


class MergeLayer(nn.Module):
    """Merge two tensors with a 1x1 convolution followed by a residual block.

    Parameters
    ----------
    channels : int or Sequence of int
        Number of channels of the first input, the second input and the output. If an
        integer is provided, it is used for all three.
    scale_initialisation : bool, default=False
        For stability, scale the output of the residual block by
        1/(channels**0.5). See VDVAE, Child 2020.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        channels: int | Sequence[int],
        scale_initialisation: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        channels : int or Sequence of int
            Number of channels of the first input, the second input and the output.
            If an integer is provided, it is used for all three.
        scale_initialisation : bool, default=False
            For stability, scale the output of the residual block by
            1/(channels**0.5). See VDVAE, Child 2020.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        if isinstance(channels, int):
            channels = [channels] * 3
        elif len(channels) == 1:
            channels = [channels[0]] * 3
        assert len(channels) == 3

        self.layer = nn.Sequential(
            Conv(channels[0] + channels[1], channels[2], 1, dimensions=dimensions),
            ResidualBlock(
                channels[2],
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            ),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Merge the two inputs.

        Parameters
        ----------
        x : torch.Tensor
            First input tensor.
        y : torch.Tensor
            Second input tensor.

        Returns
        -------
        torch.Tensor
            Merged tensor.
        """
        x = interleave(x, y)
        return self.layer(x)


class BottomUpLayer(nn.Module):
    """Bottom-up layer made of residual blocks with optional downsampling.

    Parameters
    ----------
    n_res_blocks : int
        Number of residual blocks in the layer.
    n_filters : int
        Number of filters in each residual block.
    downsampling_steps : Sequence of int
        Number of downsampling steps along each spatial axis.
    scale_initialisation : bool, default=False
        For stability, scale the output of the residual block by
        1/(channels**0.5). See VDVAE, Child 2020.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        n_res_blocks: int,
        n_filters: int,
        downsampling_steps: Sequence[int],
        scale_initialisation: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        n_res_blocks : int
            Number of residual blocks in the layer.
        n_filters : int
            Number of filters in each residual block.
        downsampling_steps : Sequence of int
            Number of downsampling steps along each spatial axis.
        scale_initialisation : bool, default=False
            For stability, scale the output of the residual block by
            1/(channels**0.5). See VDVAE, Child 2020.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()

        self.bu_blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            self.bu_blocks.append(
                ResBlockWithResampling(
                    c_in=n_filters,
                    c_out=n_filters,
                    resample="down",
                    n_resample=downsampling_steps,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bottom-up layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.bu_blocks(x)


class TopDownLayer(nn.Module):
    """Deterministic top-down layer, used by the direct denoiser.

    Merges a bottom-up skip connection and optionally a skip connection from the
    previous layer.

    Parameters
    ----------
    n_res_blocks : int
        Number of residual blocks in the layer.
    n_filters : int
        Number of filters in each residual block.
    upsampling_steps : Sequence of int
        Number of upsampling steps along each spatial axis.
    is_top_layer : bool, default=False
        Whether the layer is the top layer.
    skip : bool, default=False
        Whether to use a skip connection from the previous layer.
    scale_initialisation : bool, default=False
        For stability, scale the output of the residual block by
        1/(channels**0.5). See VDVAE, Child 2020.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        n_res_blocks: int,
        n_filters: int,
        upsampling_steps: Sequence[int],
        is_top_layer: bool = False,
        skip: bool = False,
        scale_initialisation: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        n_res_blocks : int
            Number of residual blocks in the layer.
        n_filters : int
            Number of filters in each residual block.
        upsampling_steps : Sequence of int
            Number of upsampling steps along each spatial axis.
        is_top_layer : bool, default=False
            Whether the layer is the top layer.
        skip : bool, default=False
            Whether to use a skip connection from the previous layer.
        scale_initialisation : bool, default=False
            For stability, scale the output of the residual block by
            1/(channels**0.5). See VDVAE, Child 2020.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()

        self.is_top_layer = is_top_layer
        self.skip = skip

        self.blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            self.blocks.append(
                ResBlockWithResampling(
                    n_filters,
                    n_filters,
                    resample="up",
                    n_resample=upsampling_steps,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

        if not is_top_layer:
            self.merge = MergeLayer(
                channels=n_filters,
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            )

            if skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )

    def forward(
        self,
        input_: torch.Tensor | None = None,
        skip_connection_input: torch.Tensor | None = None,
        bu_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the top-down layer.

        Parameters
        ----------
        input_ : torch.Tensor or None, default=None
            Output of the layer above, `None` for the top layer.
        skip_connection_input : torch.Tensor or None, default=None
            Skip connection from the previous layer.
        bu_value : torch.Tensor or None, default=None
            Bottom-up value at this layer.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        assert bu_value is not None
        if self.is_top_layer:
            output = bu_value
        else:
            assert input_ is not None
            output = self.merge(bu_value, input_)

        # Skip connection from previous layer
        if self.skip and not self.is_top_layer:
            assert skip_connection_input is not None
            output = self.skip_connection_merger(output, skip_connection_input)

        # Last top-down block (sequence of residual blocks)
        output = self.blocks(output)

        return output


### Ladder VAE parts
class NormalStochasticBlock(nn.Module):
    """Stochastic layer for the inference/generation path of a Ladder VAE.

    Creates p(z{i}|z{i+1}) (and q(z{i}|z{i+1}, x)) for generation (or inference).

    Parameters
    ----------
    c_in : int
        Number of input channels.
    c_vars : int
        Number of latent variable channels.
    c_out : int
        Number of output channels.
    kernel : int, default=3
        Kernel size for the convolutional layers.
    transform_p_params : bool, default=True
        Whether to compute the parameters of p(z) with a convolutional layer.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        c_in: int,
        c_vars: int,
        c_out: int,
        kernel: int = 3,
        transform_p_params: bool = True,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        c_in : int
            Number of input channels.
        c_vars : int
            Number of latent variable channels.
        c_out : int
            Number of output channels.
        kernel : int, default=3
            Kernel size for the convolutional layers.
        transform_p_params : bool, default=True
            Whether to compute the parameters of p(z) with a convolutional layer.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = Conv(
                c_in,
                2 * c_vars,
                kernel,
                padding=pad,
                padding_mode="reflect",
                dimensions=dimensions,
            )
        self.conv_in_q = Conv(
            c_in,
            2 * c_vars,
            kernel,
            padding=pad,
            padding_mode="reflect",
            dimensions=dimensions,
        )
        self.conv_out = Conv(
            c_vars,
            c_out,
            kernel,
            padding=pad,
            padding_mode="reflect",
            dimensions=dimensions,
        )

    def forward(
        self,
        p_params: torch.Tensor,
        q_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DistributionParams, DistributionParams]:
        """Sample from q(z|x) if `q_params` is given, otherwise from p(z).

        Parameters
        ----------
        p_params : torch.Tensor
            Parameters of the prior distribution p(z).
        q_params : torch.Tensor or None, default=None
            Parameters of the approximate posterior distribution q(z|x).

        Returns
        -------
        torch.Tensor
            Sample from either q(z|x) or p(z), mapped to `c_out` channels.
        dict of {str: torch.Tensor or None}
            Mean and standard deviation of q(z|x), `None` values when sampling from
            the prior.
        dict of {str: torch.Tensor or None}
            Mean and standard deviation of p(z).
        """
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_std_ = torch.chunk(p_params, dim=1, chunks=2)
        p_std = nn.functional.softplus(p_std_)

        q_mu: torch.Tensor | None
        q_std: torch.Tensor | None
        if q_params is not None:
            # Define q(z)
            q_params = self.conv_in_q(q_params)
            q_mu, q_std_ = torch.chunk(q_params, dim=1, chunks=2)
            q_std = nn.functional.softplus(q_std_)

            # Sample from q(z)
            z = torch.empty(q_mu.shape, device=q_mu.device).normal_()
            z = z * q_std + q_mu
        else:
            # Sample from p(z)
            q_mu = q_std = None
            z = torch.empty(p_mu.shape, device=p_mu.device).normal_()
            z = z * p_std + p_mu

        # Output of stochastic layer
        z = self.conv_out(z)

        q_dist: DistributionParams = {"mu": q_mu, "std": q_std}
        p_dist: DistributionParams = {"mu": p_mu, "std": p_std}
        return z, q_dist, p_dist


class VAETopDownLayer(nn.Module):
    """Top-down layer for the generative path of a Ladder VAE.

    Merges a bottom-up skip connection for the approximate posterior and optionally a
    skip connection from the previous layer.

    Parameters
    ----------
    z_dim : int
        Channels of the latent variable.
    n_res_blocks : int
        Number of residual blocks in the layer.
    n_filters : int
        Number of filters in each residual block.
    upsampling_steps : Sequence of int
        Number of upsampling steps along each spatial axis.
    is_top_layer : bool, default=False
        Whether the layer is the top layer.
    stochastic_skip : bool, default=False
        Whether to include a skip connection around the stochastic layer.
    learn_top_prior : bool, default=False
        Whether to learn the parameters of the top prior.
    top_prior_param_size : Sequence of int or None, default=None
        Size of the top prior parameters, required for the top layer.
    scale_initialisation : bool, default=False
        For stability, scale the output of the residual block by
        1/(channels**0.5). See VDVAE, Child 2020.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        z_dim: int,
        n_res_blocks: int,
        n_filters: int,
        upsampling_steps: Sequence[int],
        is_top_layer: bool = False,
        stochastic_skip: bool = False,
        learn_top_prior: bool = False,
        top_prior_param_size: Sequence[int] | None = None,
        scale_initialisation: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        z_dim : int
            Channels of the latent variable.
        n_res_blocks : int
            Number of residual blocks in the layer.
        n_filters : int
            Number of filters in each residual block.
        upsampling_steps : Sequence of int
            Number of upsampling steps along each spatial axis.
        is_top_layer : bool, default=False
            Whether the layer is the top layer.
        stochastic_skip : bool, default=False
            Whether to include a skip connection around the stochastic layer.
        learn_top_prior : bool, default=False
            Whether to learn the parameters of the top prior.
        top_prior_param_size : Sequence of int or None, default=None
            Size of the top prior parameters, required for the top layer.
        scale_initialisation : bool, default=False
            For stability, scale the output of the residual block by
            1/(channels**0.5). See VDVAE, Child 2020.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip

        if is_top_layer:
            assert top_prior_param_size is not None
            self.top_prior_params = nn.Parameter(
                torch.zeros(list(top_prior_param_size)), requires_grad=learn_top_prior
            )

        self.deterministic_blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            self.deterministic_blocks.append(
                ResBlockWithResampling(
                    n_filters,
                    n_filters,
                    resample="up",
                    n_resample=upsampling_steps,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

        self.stochastic = NormalStochasticBlock(
            c_in=n_filters,
            c_vars=z_dim,
            c_out=n_filters,
            transform_p_params=(not is_top_layer),
            dimensions=dimensions,
        )

        if not is_top_layer:
            self.merge = MergeLayer(
                channels=n_filters,
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            )

            if stochastic_skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )

    def forward(
        self,
        p_params: torch.Tensor | None = None,
        skip_connection_input: torch.Tensor | None = None,
        bu_value: torch.Tensor | None = None,
        n_img_prior: int | None = None,
    ) -> tuple[torch.Tensor, DistributionParams, DistributionParams]:
        """Apply the top-down layer.

        Parameters
        ----------
        p_params : torch.Tensor or None, default=None
            Output of the layer above, `None` for the top layer.
        skip_connection_input : torch.Tensor or None, default=None
            Skip connection from the previous layer.
        bu_value : torch.Tensor or None, default=None
            Inferred bottom-up value at this layer, `None` when sampling from the
            prior.
        n_img_prior : int or None, default=None
            Number of images to generate when sampling from the prior.

        Returns
        -------
        torch.Tensor
            Output of the layer.
        dict of {str: torch.Tensor or None}
            Mean and standard deviation of q(z|x).
        dict of {str: torch.Tensor or None}
            Mean and standard deviation of p(z).
        """
        if self.is_top_layer:
            assert p_params is None
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.repeat_interleave(n_img_prior, dim=0)
        assert p_params is not None

        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        q_params: torch.Tensor | None
        if bu_value is not None:
            if self.is_top_layer:
                q_params = bu_value
            else:
                q_params = self.merge(bu_value, p_params)
        # In generative mode, q is not used
        else:
            q_params = None

        # Samples from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None
        z, q, p = self.stochastic(
            p_params=p_params,
            q_params=q_params,
        )

        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            z = self.skip_connection_merger(z, skip_connection_input)

        z = self.deterministic_blocks(z)

        return z, q, p


### AR Decoder (PixelCNN) Parts
class ShiftedConv(nn.Module):
    """Convolutional layer with receptive field shifted left on the last dimension.

    The kernel spans only the last dimension.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    dilation : int, default=1
        Dilation rate of the convolutional kernel.
    groups : int, default=1
        Number of groups for grouped convolution.
    first : bool, default=False
        Whether this is the first layer in the network, in which case the centre
        pixel is masked.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    kernel_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        first: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the convolutional kernel.
        dilation : int, default=1
            Dilation rate of the convolutional kernel.
        groups : int, default=1
            Number of groups for grouped convolution.
        first : bool, default=False
            Whether this is the first layer in the network, in which case the centre
            pixel is masked.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.first = first

        shift = dilation * (kernel_size - 1)
        self.padding = (shift, 0)

        kernel_shape = [1] * (dimensions - 1) + [self.kernel_size]
        self.conv = Conv(
            self.in_channels,
            self.out_channels,
            kernel_shape,
            dilation=self.dilation,
            groups=self.groups,
            dimensions=dimensions,
        )

        kernel_mask = torch.ones((1, 1, *kernel_shape))
        if self.first:
            kernel_mask[..., -1] = 0
        self.register_buffer("kernel_mask", kernel_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shifted convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = F.pad(x, self.padding)
        self.conv.conv.weight.data *= self.kernel_mask
        x = self.conv(x)
        return x


class PixelCNNBlock(nn.Module):
    """Residual block for the autoregressive CNN.

    Uses `ShiftedConv` layers for the main convolutional layers and conditions on a
    tensor with the same spatial shape as the input.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    s_code_channels : int
        Number of channels in the conditioning signal code.
    kernel_size : int
        Size of the convolutional kernel.
    dilation : int, default=1
        Dilation rate of the convolutional kernel.
    groups : int, default=1
        Number of groups for grouped convolution.
    gated : bool, default=False
        Whether to use gated activations.
    first : bool, default=False
        Whether this is the first layer in the network, in which case the centre
        pixel is masked.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        s_code_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        gated: bool = False,
        first: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        s_code_channels : int
            Number of channels in the conditioning signal code.
        kernel_size : int
            Size of the convolutional kernel.
        dilation : int, default=1
            Dilation rate of the convolutional kernel.
        groups : int, default=1
            Number of groups for grouped convolution.
        gated : bool, default=False
            Whether to use gated activations.
        first : bool, default=False
            Whether this is the first layer in the network, in which case the centre
            pixel is masked.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        mid_channels = out_channels * 2 if gated else out_channels

        self.in_conv = ShiftedConv(
            in_channels,
            mid_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
            first=first,
            dimensions=dimensions,
        )
        self.s_conv = Conv(s_code_channels, mid_channels, 1, dimensions=dimensions)
        self.act_fn: Callable[[torch.Tensor], torch.Tensor]
        if gated:
            self.act_fn = _gated_activation
        else:
            self.act_fn = nn.ReLU()
        self.out_conv = Conv(
            out_channels, out_channels, 1, groups=groups, dimensions=dimensions
        )

        self.do_skip = out_channels == in_channels and not first

    def forward(
        self, x: torch.Tensor, s_code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the PixelCNN block.

        The conditioning tensor is returned unchanged to match the signature of the
        other layers used in the PixelCNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        s_code : torch.Tensor
            Conditioning tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        torch.Tensor
            Conditioning tensor.
        """
        feat = self.in_conv(x) + self.s_conv(s_code)
        feat = self.act_fn(feat)
        out = self.out_conv(feat)

        if self.do_skip:
            out = out + x

        return out, s_code


def _gated_activation(x: torch.Tensor) -> torch.Tensor:
    """Gated activation (A. Oord 2016).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with an even number of channels.

    Returns
    -------
    torch.Tensor
        Output tensor with half the number of channels.
    """
    return torch.tanh(x[:, 0::2]) * torch.sigmoid(x[:, 1::2])


class PixelCNNLayers(nn.Module):
    """Layers of the autoregressive CNN.

    Used to model spatially correlated noise in the input tensor. Noise can be
    correlated along either the x, y or z axis. The input tensor is rotated to align
    the noise with the x axis. Noise is assumed not to be correlated along the
    channel axis, which is enforced by grouped convolutions.

    Parameters
    ----------
    colour_channels : int
        Number of input channels.
    s_code_channels : int
        Number of channels in the conditioning tensor.
    kernel_size : int
        Size of the convolutional kernel.
    n_filters : int
        Number of filters in the convolutional layers.
    n_layers : int
        Number of layers in the PixelCNN.
    direction : {"x", "y", "z", "none"}, default="x"
        Axis along which noise is correlated.
    gated : bool, default=False
        Whether to use gated activations.
    checkpointed : bool, default=False
        Whether to use activation checkpointing in the forward pass.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        colour_channels: int,
        s_code_channels: int,
        kernel_size: int,
        n_filters: int,
        n_layers: int,
        direction: str = "x",
        gated: bool = False,
        checkpointed: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        colour_channels : int
            Number of input channels.
        s_code_channels : int
            Number of channels in the conditioning tensor.
        kernel_size : int
            Size of the convolutional kernel.
        n_filters : int
            Number of filters in the convolutional layers.
        n_layers : int
            Number of layers in the PixelCNN.
        direction : {"x", "y", "z", "none"}, default="x"
            Axis along which noise is correlated.
        gated : bool, default=False
            Whether to use gated activations.
        checkpointed : bool, default=False
            Whether to use activation checkpointing in the forward pass.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        self.checkpointed = checkpointed

        middle_layer = n_layers // 2 + n_layers % 2

        self.layers = nn.ModuleList()
        if direction == "y":
            self.layers.append(Rotate90(k=1, dims=[-2, -1]))
        elif direction == "z":
            self.layers.append(Rotate90(k=1, dims=[-3, -1]))
        elif direction not in ("x", "none"):
            raise ValueError(
                f"Direction {direction} not supported. Use 'x', 'y', 'z' or 'none'"
            )

        groups = colour_channels
        for i in range(n_layers):
            c_in = colour_channels if i == 0 else n_filters
            first = i == 0
            dilation = 2**i if i < middle_layer else 2 ** (n_layers - i - 1)

            self.layers.append(
                PixelCNNBlock(
                    in_channels=c_in,
                    out_channels=n_filters,
                    s_code_channels=s_code_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    gated=gated,
                    first=first,
                    dimensions=dimensions,
                )
            )
            self.layers.append(
                PixelCNNBlock(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    s_code_channels=s_code_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                    gated=gated,
                    dimensions=dimensions,
                )
            )

        if direction == "y":
            self.layers.append(Rotate90(k=-1, dims=[-2, -1]))
        elif direction == "z":
            self.layers.append(Rotate90(k=-1, dims=[-3, -1]))

    def forward(self, x: torch.Tensor, s_code: torch.Tensor) -> torch.Tensor:
        """Apply the PixelCNN layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        s_code : torch.Tensor
            Conditioning tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.checkpointed:
                x, s_code = checkpoint(  # type: ignore[misc]
                    layer,
                    x,
                    s_code,
                    use_reentrant=False,
                )
            else:
                x, s_code = layer(x, s_code)
        return x

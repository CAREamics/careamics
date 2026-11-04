"""COSDD autoregressive noise decoder.

Ported from (https://github.com/krulllab/COSDD, `models/pixelcnn.py`), MIT License,
Copyright (c) 2023 krulllab.
"""

import torch
from torch import nn

from .layers import Conv, PixelCNNLayers
from .utils import sample_mixture_model


def log_normal_pdf(
    x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Log probability density of a normal distribution.

    Parameters
    ----------
    x : torch.Tensor
        Values at which to evaluate the density.
    loc : torch.Tensor
        Mean of the distribution.
    scale : torch.Tensor
        Standard deviation of the distribution.

    Returns
    -------
    torch.Tensor
        Log probability density.
    """
    a = -torch.log(scale)
    b = torch.log(torch.tensor(2, device=loc.device) * torch.pi)
    c = ((x - loc) / scale) ** 2
    return a - 0.5 * (b + c)


class PixelCNN(nn.Module):
    """Autoregressive decoder modelling the noise as a Gaussian mixture.

    Causal convolutions are one dimensional, with shape (1, kernel_size). To
    implement vertical kernels, the input is rotated 90 degrees on entry and rotated
    back on exit.

    Parameters
    ----------
    colour_channels : int
        Number of colour channels in the target image.
    s_code_channels : int
        Number of channels in the decoded signal code.
    kernel_size : int
        Size of the kernel in the convolutional layers.
    noise_direction : {"x", "y", "z", "none"}, default="x"
        Axis along which the receptive field runs.
    n_filters : int, default=64
        Number of filters in the convolutional layers.
    n_layers : int, default=4
        Number of layers.
    n_components : int, default=5
        Number of components in the predictive Gaussian mixture model.
    gated : bool, default=False
        Whether to use gated activations (A. Oord 2016).
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
        noise_direction: str = "x",
        n_filters: int = 64,
        n_layers: int = 4,
        n_components: int = 5,
        gated: bool = False,
        checkpointed: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        colour_channels : int
            Number of colour channels in the target image.
        s_code_channels : int
            Number of channels in the decoded signal code.
        kernel_size : int
            Size of the kernel in the convolutional layers.
        noise_direction : {"x", "y", "z", "none"}, default="x"
            Axis along which the receptive field runs.
        n_filters : int, default=64
            Number of filters in the convolutional layers.
        n_layers : int, default=4
            Number of layers.
        n_components : int, default=5
            Number of components in the predictive Gaussian mixture model.
        gated : bool, default=False
            Whether to use gated activations (A. Oord 2016).
        checkpointed : bool, default=False
            Whether to use activation checkpointing in the forward pass.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        noise_direction = noise_direction.lower()
        assert noise_direction in ("x", "y", "z", "none")
        self.n_components = n_components
        self.colour_channels = colour_channels
        self.noise_direction = noise_direction

        if noise_direction == "none":
            kernel_size = 1

        # Uses grouped convolutions to ensure that each colour channel is
        # processed separately.
        if n_filters % colour_channels != 0:
            n_filters += colour_channels - (n_filters % colour_channels)

        self.layers = PixelCNNLayers(
            colour_channels=colour_channels,
            s_code_channels=s_code_channels,
            kernel_size=kernel_size,
            n_filters=n_filters,
            n_layers=n_layers,
            direction=noise_direction,
            gated=gated,
            checkpointed=checkpointed,
            dimensions=dimensions,
        )

        c_out = n_components * colour_channels * 3
        self.out_conv = Conv(
            in_channels=n_filters,
            out_channels=c_out,
            kernel_size=1,
            groups=colour_channels,
            dimensions=dimensions,
        )

    def forward(self, x: torch.Tensor, s_code: torch.Tensor) -> torch.Tensor:
        """Predict the mixture parameters of each pixel given the preceding pixels.

        Parameters
        ----------
        x : torch.Tensor
            Noisy image, shape (N, C, *spatial).
        s_code : torch.Tensor
            Latent signal code, shape (N, s_code_channels, *spatial).

        Returns
        -------
        torch.Tensor
            Mixture parameters, shape (N, 3 * n_components * C, *spatial).
        """
        out = self.layers(x, s_code)

        return self.out_conv(out)

    def extract_params(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the output of the network into the mixture parameters.

        Parameters
        ----------
        params : torch.Tensor
            Output of the network.

        Returns
        -------
        torch.Tensor
            Unnormalised log weights, mixture components along the last axis.
        torch.Tensor
            Component means, mixture components along the last axis.
        torch.Tensor
            Component pre-softplus scales, mixture components along the last axis.
        """
        logweights = params[:, 0::3].unfold(1, self.n_components, self.n_components)
        loc = params[:, 1::3].unfold(1, self.n_components, self.n_components)
        log_scale = params[:, 2::3].unfold(1, self.n_components, self.n_components)
        return logweights, loc, log_scale

    def loglikelihood(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Log likelihood of each pixel under the predicted Gaussian mixture.

        Parameters
        ----------
        x : torch.Tensor
            Noisy image, shape (N, C, *spatial).
        params : torch.Tensor
            Output of the network.

        Returns
        -------
        torch.Tensor
            Per-pixel log likelihood, shape (N, C, *spatial).
        """
        logweights, loc, log_scale = self.extract_params(params)

        scale = nn.functional.softplus(log_scale)
        logweights = nn.functional.log_softmax(logweights, dim=-1)
        log_p_per_component = log_normal_pdf(x[..., None], loc, scale)
        log_weighted_p_per_component = logweights + log_p_per_component
        log_p = torch.logsumexp(log_weighted_p_per_component, dim=-1)

        return log_p

    @torch.no_grad()
    def sample(self, s_code: torch.Tensor) -> torch.Tensor:
        """Autoregressively sample a noisy image conditioned on a signal code.

        Parameters
        ----------
        s_code : torch.Tensor
            Latent signal code, shape (N, s_code_channels, *spatial).

        Returns
        -------
        torch.Tensor
            Sampled noisy image, shape (N, C, *spatial).
        """
        image = torch.zeros(
            s_code.shape[0],
            self.colour_channels,
            *s_code.shape[2:],
            device=s_code.device,
        )

        if self.noise_direction == "x":
            for i in range(s_code.shape[-1]):
                params = self(image[..., : i + 1], s_code[..., : i + 1])
                logweights, loc, log_scale = self.extract_params(params[..., i : i + 1])
                scale = nn.functional.softplus(log_scale)
                logweights = nn.functional.log_softmax(logweights, dim=-1)
                image[..., i : i + 1] = sample_mixture_model(logweights, loc, scale)
        elif self.noise_direction == "y":
            for i in range(s_code.shape[-2]):
                params = self(image[..., : i + 1, :], s_code[..., : i + 1, :])
                logweights, loc, log_scale = self.extract_params(
                    params[..., i : i + 1, :]
                )
                scale = nn.functional.softplus(log_scale)
                logweights = nn.functional.log_softmax(logweights, dim=-1)
                image[..., i : i + 1, :] = sample_mixture_model(logweights, loc, scale)
        elif self.noise_direction == "z":
            for i in range(s_code.shape[-3]):
                params = self(image[..., : i + 1, :, :], s_code[..., : i + 1, :, :])
                logweights, loc, log_scale = self.extract_params(
                    params[..., i : i + 1, :, :]
                )
                scale = nn.functional.softplus(log_scale)
                logweights = nn.functional.log_softmax(logweights, dim=-1)
                image[..., i : i + 1, :, :] = sample_mixture_model(
                    logweights, loc, scale
                )

        return image

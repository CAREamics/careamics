"""COSDD networks factory.

Ported from the COSDD reference implementation
(https://github.com/krulllab/COSDD, `models/get_models.py`), MIT License,
Copyright (c) 2023 krulllab.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from .lvae import LadderVAE
from .pixelcnn import PixelCNN
from .s_decoder import SDecoder
from .unet import UNet

if TYPE_CHECKING:
    from careamics.config.architectures.cosdd_config import COSDDConfig

AR_DECODER_KERNEL_SIZE = 5
"""Kernel size of the autoregressive decoder."""

AR_DECODER_N_FILTERS = 64
"""Number of filters of the autoregressive decoder."""

AR_DECODER_N_LAYERS = 4
"""Number of layers of the autoregressive decoder."""


class COSDDNetworks(NamedTuple):
    """The networks making up COSDD."""

    vae: LadderVAE
    """Ladder VAE encoding the signal into a latent code."""

    ar_decoder: PixelCNN
    """Autoregressive decoder modelling the noisy image given the latent code."""

    s_decoder: SDecoder
    """Signal decoder mapping the latent code to image space."""

    direct_denoiser: UNet | None
    """Optional direct denoiser predicting the average of the signal samples."""


def get_downsampling(input_shape: Sequence[int], n_layers: int) -> list[list[int]]:
    """Compute the binary downsampling of each layer along each spatial axis.

    Each axis is halved at most `floor(log2(size)) - 1` times. Layers that do not
    downsample are spread out by removing downsampling from every other layer.

    Parameters
    ----------
    input_shape : Sequence of int
        Spatial shape of the training patches.
    n_layers : int
        Number of layers.

    Returns
    -------
    list of list of int
        Binary downsampling, of shape (n_layers, len(input_shape)).
    """
    num_halves = [math.floor(math.log2(s)) - 1 for s in input_shape]
    difference = [max(n_layers - nh, 0) for nh in num_halves]
    n_dim = len(input_shape)
    downsampling = [[1 for _ in range(n_dim)] for _ in range(n_layers)]
    for i in range(n_dim):
        j = 0
        while difference[i] > 0:
            for k in range(n_layers // 2):
                downsampling[j + k * 2][i] = 0
                difference[i] -= 1
                if difference[i] == 0:
                    break
            j += 1

    return downsampling


def create_cosdd_networks(config: COSDDConfig) -> COSDDNetworks:
    """Create the COSDD networks from the architecture configuration.

    Parameters
    ----------
    config : COSDDConfig
        COSDD architecture configuration.

    Returns
    -------
    COSDDNetworks
        The Ladder VAE, autoregressive decoder, signal decoder and optional direct
        denoiser.
    """
    downsampling = get_downsampling(config.input_shape, config.n_layers)

    lvae = LadderVAE(
        colour_channels=config.in_channels,
        img_size=config.input_shape,
        s_code_channels=config.s_code_channels,
        n_filters=config.s_code_channels,
        scale_initialisation=config.scale_initialisation,
        z_dims=config.z_dims,
        downsampling=downsampling,
        checkpointed=config.checkpointed,
        dimensions=config.conv_dims,
    )

    ar_decoder = PixelCNN(
        colour_channels=config.in_channels,
        s_code_channels=config.s_code_channels,
        kernel_size=AR_DECODER_KERNEL_SIZE,
        noise_direction=config.noise_direction,
        n_filters=AR_DECODER_N_FILTERS,
        n_layers=AR_DECODER_N_LAYERS,
        n_components=config.n_gaussians,
        dimensions=config.conv_dims,
    )

    s_decoder = SDecoder(
        colour_channels=config.in_channels,
        s_code_channels=config.s_code_channels,
        n_filters=config.s_code_channels,
        checkpointed=config.checkpointed,
        dimensions=config.conv_dims,
    )

    direct_denoiser: UNet | None = None
    if config.use_direct_denoiser:
        direct_denoiser = UNet(
            colour_channels=config.in_channels,
            n_filters=config.s_code_channels,
            n_layers=config.n_layers,
            downsampling=downsampling,
            loss_fn=config.direct_denoiser_loss,
            checkpointed=config.checkpointed,
            dimensions=config.conv_dims,
        )

    return COSDDNetworks(lvae, ar_decoder, s_decoder, direct_denoiser)

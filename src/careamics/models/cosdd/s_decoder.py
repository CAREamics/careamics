"""COSDD signal decoder.

Ported from (https://github.com/krulllab/COSDD, `models/s_decoder.py`), MIT License,
Copyright (c) 2023 krulllab.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .layers import Conv


class SDecoder(nn.Module):
    """Network mapping the latent signal code to image space.

    Parameters
    ----------
    colour_channels : int
        Number of colour channels in the target image.
    s_code_channels : int
        Number of channels in the input signal code.
    n_filters : int, default=64
        Number of filters in the convolutional layers.
    n_layers : int, default=4
        Number of layers in the network, at least 2.
    kernel_size : int, default=3
        Size of the convolutional kernel.
    checkpointed : bool, default=False
        Whether to use activation checkpointing in the forward pass.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        colour_channels: int,
        s_code_channels: int,
        n_filters: int = 64,
        n_layers: int = 4,
        kernel_size: int = 3,
        checkpointed: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        colour_channels : int
            Number of colour channels in the target image.
        s_code_channels : int
            Number of channels in the input signal code.
        n_filters : int, default=64
            Number of filters in the convolutional layers.
        n_layers : int, default=4
            Number of layers in the network, at least 2.
        kernel_size : int, default=3
            Size of the convolutional kernel.
        checkpointed : bool, default=False
            Whether to use activation checkpointing in the forward pass.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        self.checkpointed = checkpointed
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")

        self.net = nn.ModuleList()
        self.net.append(
            nn.Sequential(
                Conv(
                    s_code_channels,
                    n_filters,
                    kernel_size,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    dimensions=dimensions,
                ),
                nn.ReLU(),
            )
        )
        for _ in range(n_layers - 2):
            self.net.append(
                nn.Sequential(
                    Conv(
                        n_filters,
                        n_filters,
                        kernel_size,
                        padding=kernel_size // 2,
                        padding_mode="reflect",
                        dimensions=dimensions,
                    ),
                    nn.ReLU(),
                )
            )
        self.net.append(
            Conv(
                n_filters,
                colour_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="reflect",
                dimensions=dimensions,
            )
        )

    def forward(self, s_code: torch.Tensor) -> torch.Tensor:
        """Decode the signal code into image space.

        Parameters
        ----------
        s_code : torch.Tensor
            Latent signal code, shape (N, s_code_channels, *spatial).

        Returns
        -------
        torch.Tensor
            Decoded signal, shape (N, C, *spatial).
        """
        for i, layer in enumerate(self.net):
            if i % 2 == 0 and self.checkpointed:
                s_code = checkpoint(  # type: ignore[assignment]
                    layer,
                    s_code,
                    use_reentrant=False,
                )
            else:
                s_code = layer(s_code)
        return s_code

    @staticmethod
    def loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise mean squared error.

        Parameters
        ----------
        x : torch.Tensor
            Noisy image.
        y : torch.Tensor
            Decoded signal.

        Returns
        -------
        torch.Tensor
            Element-wise squared error.
        """
        return F.mse_loss(x, y, reduction="none")

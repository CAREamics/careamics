"""COSDD direct denoiser.

Ported from the COSDD reference implementation
(https://github.com/krulllab/COSDD, `models/unet.py`), MIT License,
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
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import BottomUpLayer, Conv, ResBlockWithResampling, TopDownLayer
from .utils import get_padded_size, spatial_pad_crop


class UNet(nn.Module):
    """Direct denoiser, a deterministic version of the Ladder VAE.

    It is trained to predict the average of the signal samples, avoiding sampling at
    prediction time.

    Parameters
    ----------
    colour_channels : int
        Number of colour channels in the input image.
    downsampling : Sequence of Sequence of int
        Binary downsampling per layer and spatial axis, of shape
        (n_layers, dimensions).
    blocks_per_layer : int, default=1
        Number of residual blocks per layer.
    n_filters : int, default=64
        Number of filters in the convolutional layers.
    n_layers : int, default=14
        Number of layers in the UNet.
    td_skip : bool, default=True
        Whether to use skip connections in the top-down pass.
    loss_fn : {"MSE", "L1"}, default="MSE"
        Loss function, the direct denoiser predicts the mean (MSE) or the
        coordinate-wise median (L1) of the signal samples.
    checkpointed : bool, default=False
        Whether to use activation checkpointing in the forward pass.
    dimensions : int, default=2
        Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        colour_channels: int,
        downsampling: Sequence[Sequence[int]],
        blocks_per_layer: int = 1,
        n_filters: int = 64,
        n_layers: int = 14,
        td_skip: bool = True,
        loss_fn: Literal["MSE", "L1"] = "MSE",
        checkpointed: bool = False,
        dimensions: int = 2,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        colour_channels : int
            Number of colour channels in the input image.
        downsampling : Sequence of Sequence of int
            Binary downsampling per layer and spatial axis, of shape
            (n_layers, dimensions).
        blocks_per_layer : int, default=1
            Number of residual blocks per layer.
        n_filters : int, default=64
            Number of filters in the convolutional layers.
        n_layers : int, default=14
            Number of layers in the UNet.
        td_skip : bool, default=True
            Whether to use skip connections in the top-down pass.
        loss_fn : {"MSE", "L1"}, default="MSE"
            Loss function, the direct denoiser predicts the mean (MSE) or the
            coordinate-wise median (L1) of the signal samples.
        checkpointed : bool, default=False
            Whether to use activation checkpointing in the forward pass.
        dimensions : int, default=2
            Dimensionality of the data (1, 2 or 3).
        """
        super().__init__()
        self.n_layers = n_layers
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.td_skip = td_skip
        if loss_fn not in ("MSE", "L1"):
            raise ValueError("UNet loss_fn should be either 'L1' or 'MSE'")
        self.loss_fn = loss_fn
        self.checkpointed = checkpointed

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
                dimensions=dimensions,
            ),
        )

        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        for i in range(self.n_layers):
            is_top = i == self.n_layers - 1

            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    dimensions=dimensions,
                )
            )

            self.top_down_layers.append(
                TopDownLayer(
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    upsampling_steps=downsampling[i],
                    skip=td_skip,
                    dimensions=dimensions,
                )
            )

        self.final_top_down = Conv(
            in_channels=n_filters,
            out_channels=colour_channels,
            kernel_size=1,
            dimensions=dimensions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the denoised image.

        Parameters
        ----------
        x : torch.Tensor
            Noisy image, shape (N, C, *spatial).

        Returns
        -------
        torch.Tensor
            Denoised image, shape (N, C, *spatial).
        """
        # Pad x to have base 2 side lengths to make resampling steps simpler
        # Save size to crop back down later
        x_size = x.size()[2:]
        padded_size = get_padded_size(x_size, self.n_downsample)
        x = spatial_pad_crop(x, padded_size)

        bu_values = self.bottomup_pass(x)

        output = self.topdown_pass(bu_values)

        # Restore original image size
        output = spatial_pad_crop(output, x_size)
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
        # Bottom-up initial layer
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

    def topdown_pass(self, bu_values: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute the top-down pass.

        Parameters
        ----------
        bu_values : Sequence of torch.Tensor
            Bottom-up values of each layer.

        Returns
        -------
        torch.Tensor
            Output of the network.
        """
        out = None
        for i in reversed(range(self.n_layers)):
            skip_input = out

            if i % 2 == 0 and self.checkpointed:
                out = checkpoint(
                    self.top_down_layers[i],
                    out,
                    skip_input,
                    bu_values[i],
                    use_reentrant=False,
                )
            else:
                out = self.top_down_layers[i](
                    out,
                    skip_input,
                    bu_values[i],
                )

        out = self.final_top_down(out)

        return out

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise loss between the target and the prediction.

        Parameters
        ----------
        x : torch.Tensor
            Target, the signal estimate of the signal decoder.
        y : torch.Tensor
            Prediction of the direct denoiser.

        Returns
        -------
        torch.Tensor
            Element-wise L1 or squared error.
        """
        if self.loss_fn == "L1":
            return F.l1_loss(x, y, reduction="none")
        return F.mse_loss(x, y, reduction="none")

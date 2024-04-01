"""
Layer module.

This submodule contains layers used in the CAREamics models.
"""

import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    """
    Convolution block used in UNets.

    Convolution block consist of two convolution layers with optional batch norm,
    dropout and with a final activation function.

    The parameters are directly mapped to PyTorch Conv2D and Conv3d parameters, see
    PyTorch torch.nn.Conv2d and torch.nn.Conv3d for more information.

    Parameters
    ----------
    conv_dim : int
        Number of dimension of the convolutions, 2 or 3.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    intermediate_channel_multiplier : int, optional
        Multiplied for the number of output channels, by default 1.
    stride : int, optional
        Stride of the convolutions, by default 1.
    padding : int, optional
        Padding of the convolutions, by default 1.
    bias : bool, optional
        Bias of the convolutions, by default True.
    groups : int, optional
        Controls the connections between inputs and outputs, by default 1.
    activation : str, optional
        Activation function, by default "ReLU".
    dropout_perc : float, optional
        Dropout percentage, by default 0.
    use_batch_norm : bool, optional
        Use batch norm, by default False.
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        out_channels: int,
        intermediate_channel_multiplier: int = 1,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        groups: int = 1,
        activation: str = "ReLU",
        dropout_perc: float = 0,
        use_batch_norm: bool = False,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimension of the convolutions, 2 or 3.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        intermediate_channel_multiplier : int, optional
            Multiplied for the number of output channels, by default 1.
        stride : int, optional
            Stride of the convolutions, by default 1.
        padding : int, optional
            Padding of the convolutions, by default 1.
        bias : bool, optional
            Bias of the convolutions, by default True.
        groups : int, optional
            Controls the connections between inputs and outputs, by default 1.
        activation : str, optional
            Activation function, by default "ReLU".
        dropout_perc : float, optional
            Dropout percentage, by default 0.
        use_batch_norm : bool, optional
            Use batch norm, by default False.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = getattr(nn, f"Conv{conv_dim}d")(
            in_channels,
            out_channels * intermediate_channel_multiplier,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        self.conv2 = getattr(nn, f"Conv{conv_dim}d")(
            out_channels * intermediate_channel_multiplier,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        self.batch_norm1 = getattr(nn, f"BatchNorm{conv_dim}d")(
            out_channels * intermediate_channel_multiplier
        )
        self.batch_norm2 = getattr(nn, f"BatchNorm{conv_dim}d")(out_channels)

        self.dropout = (
            getattr(nn, f"Dropout{conv_dim}d")(dropout_perc)
            if dropout_perc > 0
            else None
        )
        self.activation = (
            getattr(nn, f"{activation}")() if activation is not None else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.use_batch_norm:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.activation(x)
        else:
            x = self.conv1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

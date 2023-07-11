from collections import OrderedDict

import torch
import torch.nn as nn

from .layers import Conv_Block

# TODO add docstings, typing
# TODO Urgent: refactor


class UnetEncoder(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        in_channels: int = 1,
        depth: int = 3,
        num_filter_base: int = 64,
        use_batch_norm=True,
        dropout=0.0,
        pool_kernel=2,
    ) -> None:
        super().__init__()

        self.pooling = getattr(nn, f"MaxPool{conv_dim}d")(kernel_size=pool_kernel)

        enc_blocks = []

        for n in range(depth):
            out_channels = num_filter_base * (2**n)
            in_channels = in_channels if n == 0 else out_channels
            enc_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels,
                    out_channels,
                    dropout_perc=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )
            enc_blocks.append(self.pooling)

        self.encoder = nn.ModuleList(enc_blocks)

    def forward(self, x):
        for module in self.encoder:
            x = module(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        out_channels: int = 64,
        depth: int = 3,
        num_filter_base: int = 64,
        num_conv_per_depth=2,
        use_batch_norm=True,
        dropout=0.0,
    ) -> None:
        super().__init__()

        bottleneck = []

        for i in range(num_conv_per_depth - 1):
            bottleneck.append(
                Conv_Block(
                    conv_dim,
                    in_channels=out_channels,
                    out_channels=num_filter_base * 2**depth,
                    dropout_perc=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )

        self.bottleneck = nn.ModuleList(bottleneck)

    def forward(self, x):
        for module in self.bottleneck:
            x = module(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        depth: int = 3,
        num_filter_base: int = 64,
        use_batch_norm=True,
        dropout=0.0,
        last_activation=None,
    ) -> None:
        super().__init__()

        upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear" if conv_dim == 2 else "trilinear"
        )  # TODO check align_corners and mode

        dec_blocks = []
        for n in reversed(range(depth)):
            dec_blocks.append(upsampling)
            n_filter = num_filter_base * 2**n if n > 0 else num_filter_base
            dec_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels=n_filter * 2,
                    out_channels=n_filter,
                    dropout=dropout,
                    activation="ReLU" if n > 0 else last_activation,
                    use_batch_norm=use_batch_norm,
                )
            )

        self.decoder = nn.ModuleList(dec_blocks)

    def forward(features):
        skip_connections = features[::-1]
        for i, module in enumerate(self.decoder):
            x = module(features[i])
            x = torch.cat([x, skip_connections[i]])

        return x


class UNet(nn.Module):
    """UNet model. Refactored from https://github.com/juglab/n2v/blob/main/n2v/nets/unet_blocks.py

    args:
        conv_dim: int
            Dimension of the convolution layers (2 or 3)
        num_classes: int
            Number of classes to predict
        in_channels: int
            Number of input channels
        depth: int
            Number of blocks in the encoder
        num_filter_base: int
            Number of filters in the first block of the encoder
        num_conv_per_depth: int
            Number of convolutional layers per block
        activation: str
            Activation function to use
        use_batch_norm: bool
            Whether to use batch normalization
        dropout: float
            Dropout probability
        pool_kernel: int
            Kernel size of the pooling layers
        last_activation: str
            Activation function to use for the last layer

    Returns
    -------
    torch.nn.Module
        UNet model

    """

    def __init__(
        self,
        conv_dim: int,
        num_classes: int = 1,
        in_channels: int = 1,
        depth: int = 3,
        num_filter_base: int = 64,
        num_conv_per_depth=2,
        use_batch_norm=True,
        dropout=0.0,
        pool_kernel=2,
        last_activation=None,
    ) -> None:
        super().__init__()

        self.enc_blocks = UnetEncoder(
            conv_dim,
            in_channels=in_channels,
            depth=depth,
            num_filter_base=num_filter_base,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            pool_kernel=pool_kernel,
        )
        self.bottleneck = Bottleneck(
            conv_dim,
            out_channels=num_filter_base * 2**depth,
            depth=depth,
            num_filter_base=num_filter_base,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )
        self.dec_blocks = UnetDecoder(
            conv_dim,
            depth=depth,
            num_filter_base=num_filter_base,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            last_activation=last_activation,
        )
        self.final_conv = getattr(nn, f"Conv{conv_dim}d")(
            in_channels=num_filter_base * 2 ** max(0, depth - 1),
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        inputs = x.clone()
        x = self.enc_blocks(x)
        x = self.bottleneck(x)
        x = self.dec_blocks(x)
        x = self.final_conv(x)
        x = torch.add(x, inputs)
        return x

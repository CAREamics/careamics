from typing import List

import torch
import torch.nn as nn

from .layers import Conv_Block


class UnetEncoder(nn.Module):
    """Unet encoder pathway.

    Parameters
    ----------
    conv_dim : int
        controls the type of the convolution layers, 2 for 2D and 3 for 3D
    depth : int
        number of encoder blocks
    num_filter_base : int
        number of channels in the first encoder block
    use_batch_norm : bool
        whether to use batch normalization
    dropout : float
        dropout probability
    pool_kernel : int
        kernel size for the max pooling layers
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int = 1,
        depth: int = 3,
        num_filter_base: int = 64,  # TODO rename to num_channels_init
        use_batch_norm=True,
        dropout=0.0,
        pool_kernel=2,
    ) -> None:
        super().__init__()

        self.pooling = getattr(nn, f"MaxPool{conv_dim}d")(kernel_size=pool_kernel)

        encoder_blocks = []

        for n in range(depth):
            out_channels = num_filter_base * (2**n)
            in_channels = in_channels if n == 0 else out_channels // 2
            encoder_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_perc=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )
            encoder_blocks.append(self.pooling)

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        List[torch.Tensor]
            List containing the output of each encoder block(skip connections) and final
            output of the encoder
        """
        encoder_features = []
        for module in self.encoder_blocks:
            x = module(x)
            if isinstance(module, Conv_Block):
                encoder_features.append(x)
        features = [x, *encoder_features]
        return features


class UnetDecoder(nn.Module):
    """Unet decoder pathway.

    Parameters
    ----------
    conv_dim : int
        controls the type of the convolution layers, 2 for 2D and 3 for 3D
    depth : int
        number of encoder blocks
    num_filter_base : int
        number of channels in the first encoder block
    use_batch_norm : bool
        whether to use batch normalization
    dropout : float
        dropout probability
    """

    def __init__(
        self,
        conv_dim: int,
        depth: int = 3,
        num_filter_base: int = 64,
        use_batch_norm=True,
        dropout=0.0,
    ) -> None:
        super().__init__()

        upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear" if conv_dim == 2 else "trilinear"
        )  # TODO check align_corners and mode
        in_channels = out_channels = num_filter_base * 2 ** (depth - 1)
        self.bottleneck = Conv_Block(
            conv_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channel_multiplier=2,
            use_batch_norm=use_batch_norm,
            dropout_perc=dropout,
        )

        decoder_blocks = []
        for n in range(depth):
            decoder_blocks.append(upsampling)
            in_channels = num_filter_base * 2 ** (depth - n)
            out_channels = num_filter_base
            decoder_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    intermediate_channel_multiplier=2,
                    dropout_perc=dropout,
                    activation="ReLU",
                    use_batch_norm=use_batch_norm,
                )
            )

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, *features):
        """Forward pass.

        Parameters
        ----------
        features :  List[torch.Tensor]
            List containing the output of each encoder block(skip connections) and final
            output of the encoder

        Returns
        -------
        torch.Tensor
            output of the decoder
        """
        # TODO skipskipone goes brrr
        x = features[0]
        skip_connections = features[1:][::-1]
        x = self.bottleneck(x)
        for i, module in enumerate(self.decoder_blocks):
            # TODO upsample order
            x = module(x)
            if isinstance(module, nn.Upsample):
                x = torch.cat([x, skip_connections[i // 2]], axis=1)
        return x


class UNet(nn.Module):
    """UNet model. Refactored from https://github.com/juglab/n2v/blob/main/n2v/nets/unet_blocks.py.

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
        use_batch_norm=True,
        dropout=0.0,
        pool_kernel=2,
        last_activation=None,
    ) -> None:
        super().__init__()

        self.encoder = UnetEncoder(
            conv_dim,
            in_channels=in_channels,
            depth=depth,
            num_filter_base=num_filter_base,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            pool_kernel=pool_kernel,
        )

        self.decoder = UnetDecoder(
            conv_dim,
            depth=depth,
            num_filter_base=num_filter_base,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )
        self.final_conv = getattr(nn, f"Conv{conv_dim}d")(
            in_channels=num_filter_base,
            out_channels=num_classes,
            kernel_size=1,
        )
        self.last_activation = last_activation if last_activation else nn.Identity()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x :  torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            final output of the model
        """
        inputs = x.clone()
        encoder_features = self.encoder(x)
        x = self.decoder(*encoder_features)
        x = self.final_conv(x)
        x = torch.add(x, inputs)
        x = self.last_activation(x)
        return x

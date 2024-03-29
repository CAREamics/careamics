"""
UNet model.

A UNet encoder, decoder and complete model.
"""
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .layers import Conv_Block


class UnetEncoder(nn.Module):
    """
    Unet encoder pathway.

    Parameters
    ----------
    conv_dim : int
        Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
    in_channels : int, optional
        Number of input channels, by default 1.
    depth : int, optional
        Number of encoder blocks, by default 3.
    num_channels_init : int, optional
        Number of channels in the first encoder block, by default 64.
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True.
    dropout : float, optional
        Dropout probability, by default 0.0.
    pool_kernel : int, optional
        Kernel size for the max pooling layers, by default 2.
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        pool_kernel: int = 2,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
        in_channels : int, optional
            Number of input channels, by default 1.
        depth : int, optional
            Number of encoder blocks, by default 3.
        num_channels_init : int, optional
            Number of channels in the first encoder block, by default 64.
        use_batch_norm : bool, optional
            Whether to use batch normalization, by default True.
        dropout : float, optional
            Dropout probability, by default 0.0.
        pool_kernel : int, optional
            Kernel size for the max pooling layers, by default 2.
        """
        super().__init__()

        self.pooling = getattr(nn, f"MaxPool{conv_dim}d")(kernel_size=pool_kernel)

        encoder_blocks = []

        for n in range(depth):
            out_channels = num_channels_init * (2**n)
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
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        List[torch.Tensor]
            Output of each encoder block (skip connections) and final output of the
            encoder.
        """
        encoder_features = []
        for module in self.encoder_blocks:
            x = module(x)
            if isinstance(module, Conv_Block):
                encoder_features.append(x)
        features = [x, *encoder_features]
        return features


class UnetDecoder(nn.Module):
    """
    Unet decoder pathway.

    Parameters
    ----------
    conv_dim : int
        Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
    depth : int, optional
        Number of decoder blocks, by default 3.
    num_channels_init : int, optional
        Number of channels in the first encoder block, by default 64.
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True.
    dropout : float, optional
        Dropout probability, by default 0.0.
    """

    def __init__(
        self,
        conv_dim: int,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
        depth : int, optional
            Number of decoder blocks, by default 3.
        num_channels_init : int, optional
            Number of channels in the first encoder block, by default 64.
        use_batch_norm : bool, optional
            Whether to use batch normalization, by default True.
        dropout : float, optional
            Dropout probability, by default 0.0.
        """
        super().__init__()

        upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear" if conv_dim == 2 else "trilinear"
        )
        in_channels = out_channels = num_channels_init * 2 ** (depth - 1)
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
            in_channels = num_channels_init * 2 ** (depth - n)
            out_channels = num_channels_init
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

    def forward(self, *features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        *features :  List[torch.Tensor]
            List containing the output of each encoder block(skip connections) and final
            output of the encoder.

        Returns
        -------
        torch.Tensor
            Output of the decoder.
        """
        x = features[0]
        skip_connections = features[1:][::-1]
        x = self.bottleneck(x)
        for i, module in enumerate(self.decoder_blocks):
            x = module(x)
            if isinstance(module, nn.Upsample):
                x = torch.cat([x, skip_connections[i // 2]], axis=1)
        return x


class UNet(nn.Module):
    """
    UNet model.

    Adapted for PyTorch from
    https://github.com/juglab/n2v/blob/main/n2v/nets/unet_blocks.py.

    Parameters
    ----------
    conv_dim : int
        Number of dimensions of the convolution layers (2 or 3).
    num_classes : int, optional
        Number of classes to predict, by default 1.
    in_channels : int, optional
        Number of input channels, by default 1.
    depth : int, optional
        Number of downsamplings, by default 3.
    num_channels_init : int, optional
        Number of filters in the first convolution layer, by default 64.
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True.
    dropout : float, optional
        Dropout probability, by default 0.0.
    pool_kernel : int, optional
        Kernel size of the pooling layers, by default 2.
    last_activation : Optional[Callable], optional
        Activation function to use for the last layer, by default None.
    """

    def __init__(
        self,
        conv_dim: int,
        num_classes: int = 1,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        pool_kernel: int = 2,
        last_activation: Optional[Callable] = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimensions of the convolution layers (2 or 3).
        num_classes : int, optional
            Number of classes to predict, by default 1.
        in_channels : int, optional
            Number of input channels, by default 1.
        depth : int, optional
            Number of downsamplings, by default 3.
        num_channels_init : int, optional
            Number of filters in the first convolution layer, by default 64.
        use_batch_norm : bool, optional
            Whether to use batch normalization, by default True.
        dropout : float, optional
            Dropout probability, by default 0.0.
        pool_kernel : int, optional
            Kernel size of the pooling layers, by default 2.
        last_activation : Optional[Callable], optional
            Activation function to use for the last layer, by default None.
        """
        super().__init__()

        self.encoder = UnetEncoder(
            conv_dim,
            in_channels=in_channels,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            pool_kernel=pool_kernel,
        )

        self.decoder = UnetDecoder(
            conv_dim,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )
        self.final_conv = getattr(nn, f"Conv{conv_dim}d")(
            in_channels=num_channels_init,
            out_channels=num_classes,
            kernel_size=1,
        )
        self.last_activation = last_activation if last_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x :  torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the model.
        """
        encoder_features = self.encoder(x)
        x = self.decoder(*encoder_features)
        x = self.final_conv(x)
        x = self.last_activation(x)
        return x

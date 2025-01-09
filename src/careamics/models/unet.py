"""
UNet model.

A UNet encoder, decoder and complete model.
"""

from typing import Any, Union

import torch
import torch.nn as nn

from ..config.support import SupportedActivation
from .activation import get_activation
from .layers import Conv_Block, MaxBlurPool


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
    n2v2 : bool, optional
        Whether to use N2V2 architecture, by default False.
    groups : int, optional
        Number of blocked connections from input channels to output
        channels, by default 1.
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
        n2v2: bool = False,
        groups: int = 1,
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
        n2v2 : bool, optional
            Whether to use N2V2 architecture, by default False.
        groups : int, optional
            Number of blocked connections from input channels to output
            channels, by default 1.
        """
        super().__init__()

        self.pooling = (
            getattr(nn, f"MaxPool{conv_dim}d")(kernel_size=pool_kernel)
            if not n2v2
            else MaxBlurPool(dim=conv_dim, kernel_size=3, max_pool_size=pool_kernel)
        )

        encoder_blocks = []

        for n in range(depth):
            out_channels = num_channels_init * (2**n) * groups
            in_channels = in_channels if n == 0 else out_channels // 2
            encoder_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_perc=dropout,
                    use_batch_norm=use_batch_norm,
                    groups=groups,
                )
            )
            encoder_blocks.append(self.pooling)
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        list[torch.Tensor]
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
    n2v2 : bool, optional
        Whether to use N2V2 architecture, by default False.
    groups : int, optional
        Number of blocked connections from input channels to output
        channels, by default 1.
    """

    def __init__(
        self,
        conv_dim: int,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        n2v2: bool = False,
        groups: int = 1,
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
        n2v2 : bool, optional
            Whether to use N2V2 architecture, by default False.
        groups : int, optional
            Number of blocked connections from input channels to output
            channels, by default 1.
        """
        super().__init__()

        upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear" if conv_dim == 2 else "trilinear"
        )
        in_channels = out_channels = num_channels_init * groups * (2 ** (depth - 1))

        self.n2v2 = n2v2
        self.groups = groups

        self.bottleneck = Conv_Block(
            conv_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channel_multiplier=2,
            use_batch_norm=use_batch_norm,
            dropout_perc=dropout,
            groups=self.groups,
        )

        decoder_blocks: list[nn.Module] = []
        for n in range(depth):
            decoder_blocks.append(upsampling)
            in_channels = (num_channels_init * 2 ** (depth - n)) * groups
            out_channels = in_channels // 2
            decoder_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels=(
                        in_channels + in_channels // 2 if n > 0 else in_channels
                    ),
                    out_channels=out_channels,
                    intermediate_channel_multiplier=2,
                    dropout_perc=dropout,
                    activation="ReLU",
                    use_batch_norm=use_batch_norm,
                    groups=groups,
                )
            )

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        *features :  list[torch.Tensor]
            List containing the output of each encoder block(skip connections) and final
            output of the encoder.

        Returns
        -------
        torch.Tensor
            Output of the decoder.
        """
        x: torch.Tensor = features[0]
        skip_connections: tuple[torch.Tensor, ...] = features[-1:0:-1]

        x = self.bottleneck(x)

        for i, module in enumerate(self.decoder_blocks):
            x = module(x)
            if isinstance(module, nn.Upsample):
                # divide index by 2 because of upsampling layers
                skip_connection: torch.Tensor = skip_connections[i // 2]
                if self.n2v2:
                    if x.shape != skip_connections[-1].shape:
                        x = self._interleave(x, skip_connection, self.groups)
                else:
                    x = self._interleave(x, skip_connection, self.groups)
        return x

    @staticmethod
    def _interleave(A: torch.Tensor, B: torch.Tensor, groups: int) -> torch.Tensor:
        """Interleave two tensors.

        Splits the tensors `A` and `B` into equally sized groups along the channel
        axis (axis=1); then concatenates the groups in alternating order along the
        channel axis, starting with the first group from tensor A.

        Parameters
        ----------
        A : torch.Tensor
            First tensor.
        B : torch.Tensor
            Second tensor.
        groups : int
            The number of groups.

        Returns
        -------
        torch.Tensor
            Interleaved tensor.

        Raises
        ------
        ValueError:
            If either of `A` or `B`'s channel axis is not divisible by `groups`.
        """
        if (A.shape[1] % groups != 0) or (B.shape[1] % groups != 0):
            raise ValueError(f"Number of channels not divisible by {groups} groups.")

        m = A.shape[1] // groups
        n = B.shape[1] // groups

        A_groups: list[torch.Tensor] = [
            A[:, i * m : (i + 1) * m] for i in range(groups)
        ]
        B_groups: list[torch.Tensor] = [
            B[:, i * n : (i + 1) * n] for i in range(groups)
        ]

        interleaved = torch.cat(
            [
                tensor_list[i]
                for i in range(groups)
                for tensor_list in [A_groups, B_groups]
            ],
            dim=1,
        )

        return interleaved


class UNet(nn.Module):
    """
    UNet model.

    Adapted for PyTorch from:
    https://github.com/juglab/n2v/blob/main/n2v/nets/unet_blocks.py.

    Parameters
    ----------
    conv_dims : int
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
    final_activation : Optional[Callable], optional
        Activation function to use for the last layer, by default None.
    n2v2 : bool, optional
        Whether to use N2V2 architecture, by default False.
    independent_channels : bool
        Whether to train the channels independently, by default True.
    **kwargs : Any
        Additional keyword arguments, unused.
    """

    def __init__(
        self,
        conv_dims: int,
        num_classes: int = 1,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        pool_kernel: int = 2,
        final_activation: Union[SupportedActivation, str] = SupportedActivation.NONE,
        n2v2: bool = False,
        independent_channels: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dims : int
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
        final_activation : Optional[Callable], optional
            Activation function to use for the last layer, by default None.
        n2v2 : bool, optional
            Whether to use N2V2 architecture, by default False.
        independent_channels : bool
            Whether to train parallel independent networks for each channel, by
            default True.
        **kwargs : Any
            Additional keyword arguments, unused.
        """
        super().__init__()

        groups = in_channels if independent_channels else 1

        self.encoder = UnetEncoder(
            conv_dims,
            in_channels=in_channels,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            pool_kernel=pool_kernel,
            n2v2=n2v2,
            groups=groups,
        )

        self.decoder = UnetDecoder(
            conv_dims,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            n2v2=n2v2,
            groups=groups,
        )
        self.final_conv = getattr(nn, f"Conv{conv_dims}d")(
            in_channels=num_channels_init * groups,
            out_channels=num_classes,
            kernel_size=1,
            groups=groups,
        )
        self.final_activation = get_activation(final_activation)

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
        x = self.final_activation(x)
        return x

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import List
from torch.nn import init

from .layers import DownConv, UpConv, conv1x1, Conv_Block_tf, BlurPool2d, BlurPool3d


# TODO add docstings, typing


class UP_MODE(str, Enum):
    """Up convolution mode"""

    TRANSPOSE = "transpose"
    UPSAMPLE = "upsample"

    @staticmethod
    def list() -> List[str]:
        """List available up convolution modes

        Returns
        -------
        List[str]
            List of available up convolution modes
        """
        return [mode.value for mode in UP_MODE]


class MERGE_MODE(str, Enum):
    """Merge mode"""

    CONCAT = "concat"
    ADD = "add"

    @staticmethod
    def list() -> List[str]:
        """List available up merge modes

        Returns
        -------
        List[str]
            List of available up merge modes
        """
        return [mode.value for mode in MERGE_MODE]


class UNet(nn.Module):
    """`UNet` class is based on Ronneberger et al. 2015 [1].

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')

    References:
        [1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional
        networks for biomedical image segmentation. In MICCAI 2015, 2015, Proceedings,
        Part III 18 (pp. 234-241). Springer International Publishing.
    """

    def __init__(
        self,
        conv_dim: int,
        num_classes: int = 1,
        in_channels: int = 1,
        depth: int = 3,
        num_filter_base: int = 64,
        up_mode: str = "upsample",
        merge_mode: str = "concat",
        n2v2: bool = False,
    ):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super().__init__()

        if depth < 1:
            raise ValueError(f"depth must be greater than 0 (got {depth}).")

        if up_mode.lower() in UP_MODE.list():
            self.up_mode = up_mode.lower()  # TODO replace by enum, also in layers.py
        else:
            raise ValueError(
                f"{up_mode} is not a valid mode for "
                f"upsampling. Only {UP_MODE.list()} are allowed."
            )

        if merge_mode.lower() in MERGE_MODE.list():
            self.merge_mode = (
                merge_mode.lower()
            )  # TODO replace by enum, also in layers.py
        else:
            raise ValueError(
                f"{merge_mode} is not a valid mode for "
                f"merging up and down paths. Only {MERGE_MODE.list()} are allowed."
            )

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                'up_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )

        self.num_classes = num_classes
        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.num_filter_base = num_filter_base
        self.depth = depth
        self.n2v2 = n2v2

        self.down_convs = []
        self.up_convs = []

        self.noiseSTD = nn.Parameter(data=torch.log(torch.tensor(0.5)))

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.num_filter_base * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(
                self.conv_dim, ins, outs, pooling=pooling, n2v2=self.n2v2
            )
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(
                self.conv_dim,
                ins,
                outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                skip=(i == depth - 2),
            )
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(self.conv_dim, outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x


class UNet_tf(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        num_classes: int = 1,
        in_channels: int = 1,
        depth: int = 3,
        num_filter_base: int = 64,
        num_conv_per_depth=2,
        activation="ReLU",
        use_batch_norm=True,
        dropout=0.0,
        pool_kernel=2,
        last_activation=None,
        n2v2: bool = False,
        skip_skipone=False,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.num_conv_per_depth = num_conv_per_depth
        self.conv_block_in = Conv_Block_tf
        self.conv_block = Conv_Block_tf

        if n2v2:
            self.pooling = (
                BlurPool2D if conv_dim == 2 else BlurPool3D
            )  # TODO getattr from layers
        else:
            self.pooling = getattr(nn, f"MaxPool{conv_dim}d")(kernel_size=pool_kernel)

        self.upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear"
        )  # TODO check align_corners and mode
        self.skipone = (
            skip_skipone  # TODO whoever called this skip_skipone must explain himself
        )
        enc_blocks = OrderedDict()
        bottleneck = OrderedDict()
        dec_blocks = OrderedDict()
        self.skip_layers_ouputs = OrderedDict()

        # TODO implements better layer naming
        # Encoder
        for n in range(self.depth):
            for i in range(self.num_conv_per_depth):
                in_channels = in_channels if i == 0 else out_channels
                out_channels = num_filter_base * (2**n)
                layer = self.conv_block(
                    conv_dim,
                    in_channels,
                    out_channels,
                    stride=1,
                    padding=1,
                    bias=True,
                    groups=1,
                    activation="ReLU",
                    dropout_perc=0,
                    use_batch_norm=use_batch_norm,
                )
                enc_blocks[f"encoder_conv_d{n}_num{i}"] = layer

            if skip_skipone:
                if n > 0:
                    enc_blocks[f"skip_encoder_conv_d{n}"] = enc_blocks.pop(
                        f"encoder_conv_d{n}_num{i}"
                    )
            else:
                enc_blocks[f"skip_encoder_conv_d{n}"] = enc_blocks.pop(
                    f"encoder_conv_d{n}_num{i}"
                )

            enc_blocks[f"encoder_pool_d{n}"] = self.pooling

        # Bottleneck
        for i in range(num_conv_per_depth - 1):
            bottleneck[f"bottleneck_num{i}"] = self.conv_block(
                conv_dim,
                in_channels=out_channels,
                out_channels=num_filter_base * 2**depth,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
                activation="ReLU",
                dropout_perc=0,
                use_batch_norm=use_batch_norm,
            )

        bottleneck["bottleneck_final"] = self.conv_block(
            conv_dim,
            in_channels=num_filter_base * 2**depth,
            out_channels=num_filter_base * 2 ** max(0, depth - 1),
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        # Decoder
        for n in reversed(range(depth)):
            dec_blocks[f"upsampling_d{n}"] = self.upsampling
            for i in range(num_conv_per_depth - 1):
                n_filter = num_filter_base * 2**n if n > 0 else num_filter_base
                dec_blocks[f"decoder_conv_d{n}_num{i}"] = self.conv_block(
                    conv_dim,
                    in_channels=n_filter * 2,
                    out_channels=n_filter,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )

            dec_blocks[f"decoder_conv_d{n}"] = self.conv_block(
                conv_dim,
                in_channels=n_filter,
                out_channels=num_filter_base * 2 ** max(0, n - 1),
                dropout=dropout,
                activation=activation if n > 0 else last_activation,
                use_batch_norm=use_batch_norm,
            )

        self.enc_blocks = nn.ModuleDict(enc_blocks)
        self.bottleneck = nn.ModuleDict(bottleneck)
        self.dec_blocks = nn.ModuleDict(dec_blocks)
        self.final_conv = getattr(nn, f"Conv{conv_dim}d")(
            in_channels=num_filter_base * 2 ** max(0, n - 1),
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        # TODO certain input sizes lead to shape mismatch, eg 160x150
        inputs = x.clone()
        for module_name in self.enc_blocks:
            x = self.enc_blocks[module_name](x)
            if module_name.startswith("skip"):
                self.skip_layers_ouputs[module_name] = x

        for module_name in self.bottleneck:
            x = self.bottleneck[module_name](x)

        for module_name in self.dec_blocks:
            if module_name.startswith("upsampling"):
                x = self.dec_blocks[module_name](x)
                skip_connection = self.skip_layers_ouputs[
                    module_name.replace("upsampling", "skip_encoder_conv")
                ]
                x = torch.cat((x, skip_connection), axis=1)
            else:
                x = self.dec_blocks[module_name](x)
        x = self.final_conv(x)
        x = torch.add(x, inputs)
        return x

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn import init

from .layers import DownConv, UpConv, conv1x1


# TODO add docstings, typing


class UNet(nn.Module):
    """`UNet` class is based on https://arxiv.org/abs/1505.04597
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
    """

    def __init__(
        self,
        conv_dim,
        num_classes=1,
        in_channels=1,
        depth=5,
        start_filts=64,
        up_mode="transpose",
        merge_mode="add",
        n2v2=False,
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

        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for '
                'upsampling. Only "transpose" and '
                '"upsample" are allowed.'.format(up_mode)
            )

        if merge_mode in ("concat", "add"):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for'
                "merging up and down paths. "
                'Only "concat" and '
                '"add" are allowed.'.format(up_mode)
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
        self.start_filts = start_filts
        self.depth = depth
        self.n2v2 = n2v2

        self.down_convs = []
        self.up_convs = []

        self.noiseSTD = nn.Parameter(data=torch.log(torch.tensor(0.5)))

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
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

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.

        x = self.conv_final(x)

        return x

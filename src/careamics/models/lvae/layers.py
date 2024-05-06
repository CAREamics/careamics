"""
Script containing the common layers (nn.Module) reused by the LadderVAE architecture.
"""
import torch
import torch.nn as nn



class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.
    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (batchnorm, activation, conv, dropout).
    """

    default_kernel_size = (3, 3)

    def __init__(self,
                 channels: int,
                 nonlin,
                 kernel=None,
                 groups=1,
                 batchnorm: bool = True,
                 block_type: str = None,
                 dropout=None,
                 gated=None,
                 skip_padding=False,
                 conv2d_bias=True):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        self.skip_padding = skip_padding
        pad = [0] * len(kernel) if self.skip_padding else [k // 2 for k in kernel]
        print(kernel, pad)
        self.gated = gated
        modules = []

        if block_type == 'cabdcabd':
            for i in range(2):
                conv = nn.Conv2d(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                modules.append(nonlin())
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                if dropout is not None:
                    modules.append(nn.Dropout2d(dropout))

        elif block_type == 'bacdbac':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin())
                conv = nn.Conv2d(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(nn.Dropout2d(dropout))

        elif block_type == 'bacdbacd':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin())
                conv = nn.Conv2d(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                modules.append(nn.Dropout2d(dropout))

        else:
            raise ValueError("unrecognized block type '{}'".format(block_type))

        # if gated:
        #     modules.append(GateLayer2d(channels, 1, nonlin))
        # self.block = nn.Sequential(*modules)

    def forward(self, x):

        out = self.block(x)
        if out.shape != x.shape:
            return out + F.center_crop(x, out.shape[-2:])
        else:
            return out + x


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling steps (each by a factor of 2).
    The mode can be top-down or bottom-up, and the block does up- and
    down-sampling by a factor of 2, respectively. Resampling is performed at
    the beginning of the block, through strided convolution.
    The number of channels is adjusted at the beginning and end of the block,
    through convolutional layers with kernel size 1. The number of internal
    channels is by default the same as the number of output channels, but
    min_inner_channels overrides this behaviour.
    Other parameters: kernel size, nonlinearity, and groups of the internal
    residual block; whether batch normalization and dropout are performed;
    whether the residual path has a gate layer at the end. There are a few
    residual block structures to choose from.
    """

    def __init__(self,
                 mode,
                 c_in,
                 c_out,
                 nonlin=nn.LeakyReLU,
                 resample=False,
                 res_block_kernel=None,
                 groups=1,
                 batchnorm=True,
                 res_block_type=None,
                 dropout=None,
                 min_inner_channels=None,
                 gated=None,
                 lowres_input=False,
                 skip_padding=False,
                 conv2d_bias=True):
        super().__init__()
        assert mode in ['top-down', 'bottom-up']
        if min_inner_channels is None:
            min_inner_channels = 0
        inner_filters = max(c_out, min_inner_channels)

        # Define first conv layer to change channels and/or up/downsample
        if resample:
            if mode == 'bottom-up':  # downsample
                self.pre_conv = nn.Conv2d(in_channels=c_in,
                                          out_channels=inner_filters,
                                          kernel_size=3,
                                          padding=1,
                                          stride=2,
                                          groups=groups,
                                          bias=conv2d_bias)
            elif mode == 'top-down':  # upsample
                self.pre_conv = nn.ConvTranspose2d(in_channels=c_in,
                                                   out_channels=inner_filters,
                                                   kernel_size=3,
                                                   padding=1,
                                                   stride=2,
                                                   groups=groups,
                                                   output_padding=1,
                                                   bias=conv2d_bias)
        elif c_in != inner_filters:
            self.pre_conv = nn.Conv2d(c_in, inner_filters, 1, groups=groups, bias=conv2d_bias)
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_filters,
            nonlin=nonlin,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            dropout=dropout,
            gated=gated,
            block_type=res_block_type,
            skip_padding=skip_padding,
            conv2d_bias=conv2d_bias,
        )
        # Define last conv layer to get correct num output channels
        if inner_filters != c_out:
            self.post_conv = nn.Conv2d(inner_filters, c_out, 1, groups=groups, bias=conv2d_bias)
        else:
            self.post_conv = None

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        x = self.res(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, upsample=False, **kwargs):
        kwargs['resample'] = upsample
        super().__init__('top-down', *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, downsample=False, **kwargs):
        kwargs['resample'] = downsample
        super().__init__('bottom-up', *args, **kwargs)

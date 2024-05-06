"""
Script containing the layers (nn.Module) used by the LadderVAE architecture.
"""

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

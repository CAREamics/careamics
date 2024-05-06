"""
Script containing the common layers (nn.Module) reused by the LadderVAE architecture.
"""
import torch
import torch.nn as nn
from typing import Union



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

    def __init__(
        self,
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
        conv2d_bias=True
    ):
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

class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer for inference, roughly the same as the
    small deterministic Resnet in top-down layers. Consists of a sequence of
    bottom-up deterministic residual blocks with downsampling.
    """

    def __init__(
        self,
        n_filters: int,
        n_res_blocks: int,
        downsampling_steps: int = 0,
        nonlin=None,
        batchnorm: bool = True,
        dropout: Union[None, float] = None,
        res_block_type: str = None,
        res_block_kernel: int = None,
        res_block_skip_padding: bool = False,
        gated: bool = None,
        multiscale_lowres_size_factor: int = None,
        enable_multiscale: bool = False,
        lowres_separate_branch=False,
        multiscale_retain_spatial_dims: bool = False,
        decoder_retain_spatial_dims: bool = False,
        output_expected_shape=None
    ):
        """
        Args:
            n_res_blocks: Number of BottomUpDeterministicResBlock blocks present in this layer.
            n_filters:      Number of channels which is present through out this layer.
            downsampling_steps: How many times downsampling has to be done in this layer. This is typically 1.
            nonlin: What non linear activation is to be applied at various places in this module.
            batchnorm: Whether to apply batch normalization at various places or not.
            dropout: Amount of dropout to be applied at various places.
            res_block_type: Example: 'bacdbac'. It has the constitution of the residual block.
            gated: This is also an argument for the residual block. At the end of residual block, whether 
            there should be a gate or not.
            res_block_kernel:int => kernel size for the residual blocks in the bottom up layer.
            multiscale_lowres_size_factor: How small is the bu_value when compared with low resolution tensor.
            enable_multiscale: Whether to enable multiscale or not.
            multiscale_retain_spatial_dims: typically the output of the bottom-up layer scales down spatially.
                                            However, with this set, we return the same spatially sized tensor.
            output_expected_shape: What should be the shape of the output of this layer. Only used if enable_multiscale is True.
        """
        super().__init__()
        self.enable_multiscale = enable_multiscale
        self.lowres_separate_branch = lowres_separate_branch
        self.multiscale_retain_spatial_dims = multiscale_retain_spatial_dims
        self.output_expected_shape = output_expected_shape
        self.decoder_retain_spatial_dims = decoder_retain_spatial_dims
        assert self.output_expected_shape is None or self.enable_multiscale is True

        bu_blocks_downsized = []
        bu_blocks_samesize = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1
            block = BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                nonlin=nonlin,
                downsample=do_resample,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                skip_padding=res_block_skip_padding,
                gated=gated,
            )
            if do_resample:
                bu_blocks_downsized.append(block)
            else:
                bu_blocks_samesize.append(block)

        self.net_downsized = nn.Sequential(*bu_blocks_downsized)
        self.net = nn.Sequential(*bu_blocks_samesize)
        # using the same net for the lowresolution (and larger sized image)
        self.lowres_net = self.lowres_merge = self.multiscale_lowres_size_factor = None
        if self.enable_multiscale:
            self._init_multiscale(
                n_filters=n_filters,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                multiscale_retain_spatial_dims=multiscale_retain_spatial_dims,
                multiscale_lowres_size_factor=multiscale_lowres_size_factor,
            )

        msg = f'[{self.__class__.__name__}] McEnabled:{int(enable_multiscale)} '
        if enable_multiscale:
            msg += f'McParallelBeam:{int(multiscale_retain_spatial_dims)} McFactor{multiscale_lowres_size_factor}'
        print(msg)

    def _init_multiscale(self,
                         n_filters=None,
                         nonlin=None,
                         batchnorm=None,
                         dropout=None,
                         res_block_type=None,
                         multiscale_retain_spatial_dims=None,
                         multiscale_lowres_size_factor=None):
        self.multiscale_lowres_size_factor = multiscale_lowres_size_factor
        self.lowres_net = self.net
        if self.lowres_separate_branch:
            self.lowres_net = deepcopy(self.net)

        self.lowres_merge = MergeLowRes(
            channels=n_filters,
            merge_type='residual',
            nonlin=nonlin,
            batchnorm=batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            multiscale_retain_spatial_dims=multiscale_retain_spatial_dims,
            multiscale_lowres_size_factor=self.multiscale_lowres_size_factor,
        )

    def forward(self, x, lowres_x=None):
        primary_flow = self.net_downsized(x)
        primary_flow = self.net(primary_flow)

        if self.enable_multiscale is False:
            assert lowres_x is None
            return primary_flow, primary_flow

        if lowres_x is not None:
            lowres_flow = self.lowres_net(lowres_x)
            merged = self.lowres_merge(primary_flow, lowres_flow)
        else:
            merged = primary_flow

        if self.multiscale_retain_spatial_dims is False or self.decoder_retain_spatial_dims is True:
            return merged, merged

        if self.output_expected_shape is not None:
            expected_shape = self.output_expected_shape
        else:
            fac = self.multiscale_lowres_size_factor
            expected_shape = (merged.shape[-2] // fac, merged.shape[-1] // fac)
            assert merged.shape[-2:] != expected_shape

        value_to_use_in_topdown = crop_img_tensor(merged, expected_shape)
        return merged, value_to_use_in_topdown

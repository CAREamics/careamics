"""Script containing the common basic blocks (nn.Module) reused by the LadderVAE."""

from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .stochastic import NormalStochasticBlock
from .utils import (
    crop_img_tensor,
    pad_img_tensor,
)

ConvType = Union[nn.Conv2d, nn.Conv3d]
NormType = Union[nn.BatchNorm2d, nn.BatchNorm3d]
DropoutType = Union[nn.Dropout2d, nn.Dropout3d]


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.

    Some architectural notes:
        - The number of input, intermediate, and output channels is the same,
        - Padding is always 'same',
        - The 2 convolutional layers have the same groups,
        - No stride allowed,
        - Kernel sizes must be odd.

    The output isgiven by: `out = gate(f(x)) + x`.
    The presence of the gating mechanism is optional, and f(x) has different
    structures depending on the `block_type` argument.
    Specifically, `block_type` is a string specifying the block's structure, with:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, "bacdbacd" defines a block with 2x[batchnorm, activation, conv, dropout].
    """

    default_kernel_size = (3, 3)

    def __init__(
        self,
        channels: int,
        nonlin: Callable,
        conv_strides: tuple[int] = (2, 2),
        kernel: Union[int, Iterable[int], None] = None,
        groups: int = 1,
        batchnorm: bool = True,
        block_type: str = None,
        dropout: float = None,
        gated: bool = None,
        conv2d_bias: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        channels: int
            The number of input and output channels (they are the same).
        nonlin: Callable
            The non-linearity function used in the block (e.g., `nn.ReLU`).
        kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        groups: int, optional
            The number of groups to consider in the convolutions. Default is 1.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        block_type: str, optional
            A string specifying the block structure, check class docstring for more info.
            Default is `None`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        gated: bool, optional
            Whether to use gated layer. Default is `None`.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        """
        super().__init__()

        # Set kernel size & padding
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all(k % 2 == 1 for k in kernel), "kernel sizes have to be odd"
        kernel = list(kernel)

        # Define modules
        conv_layer: ConvType = getattr(nn, f"Conv{len(conv_strides)}d")
        norm_layer: NormType = getattr(nn, f"BatchNorm{len(conv_strides)}d")
        dropout_layer: DropoutType = getattr(nn, f"Dropout{len(conv_strides)}d")
        # TODO: same comment as in lvae.py, would be more readable to have `conv_dims`

        modules = []
        if block_type == "cabdcabd":
            for i in range(2):
                conv = conv_layer(
                    channels,
                    channels,
                    kernel[i],
                    padding="same",
                    groups=groups,
                    bias=conv2d_bias,
                )
                modules.append(conv)
                modules.append(nonlin)
                if batchnorm:
                    modules.append(norm_layer(channels))
                if dropout is not None:
                    modules.append(dropout_layer(dropout))
        elif block_type == "bacdbac":
            for i in range(2):
                if batchnorm:
                    modules.append(norm_layer(channels))
                modules.append(nonlin)
                conv = conv_layer(
                    channels,
                    channels,
                    kernel[i],
                    padding="same",
                    groups=groups,
                    bias=conv2d_bias,
                )
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(dropout_layer(dropout))
        elif block_type == "bacdbacd":
            for i in range(2):
                if batchnorm:
                    modules.append(norm_layer(channels))
                modules.append(nonlin)
                conv = conv_layer(
                    channels,
                    channels,
                    kernel[i],
                    padding="same",
                    groups=groups,
                    bias=conv2d_bias,
                )
                modules.append(conv)
                modules.append(dropout_layer(dropout))

        else:
            raise ValueError(f"unrecognized block type '{block_type}'")

        self.gated = gated
        if gated:
            modules.append(
                GateLayer(
                    channels=channels,
                    conv_strides=conv_strides,
                    kernel_size=1,
                    nonlin=nonlin,
                )
            )

        self.block = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input tensor # TODO add shape

        Returns
        -------
        torch.Tensor
            output tensor # TODO add shape
        """
        out = self.block(x)
        assert (
            out.shape == x.shape
        ), f"output shape: {out.shape} != input shape: {x.shape}"
        return out + x


class ResidualGatedBlock(ResidualBlock):
    """Layer class that implements a residual block with a gating mechanism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gated=True)


class GateLayer(nn.Module):
    """
    Layer class that implements a gating mechanism.

    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(
        self,
        channels: int,
        conv_strides: tuple[int] = (2, 2),
        kernel_size: int = 3,
        nonlin: Callable = nn.LeakyReLU(),
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        conv_layer: ConvType = getattr(nn, f"Conv{len(conv_strides)}d")
        self.conv = conv_layer(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nonlin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input # TODO add shape

        Returns
        -------
        torch.Tensor
            output # TODO add shape
        """
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate


class ResBlockWithResampling(nn.Module):
    """
    Residual block with resampling.

    Residual block that takes care of resampling (i.e. downsampling or upsampling) steps (by a factor 2).
    It is structured as follows:
        1. `pre_conv`: a downsampling or upsampling strided convolutional layer in case of resampling, or
            a 1x1 convolutional layer that maps the number of channels of the input to `inner_channels`.
        2. `ResidualBlock`
        3. `post_conv`: a 1x1 convolutional layer that maps the number of channels to `c_out`.

    Some implementation notes:
    - Resampling is performed through a strided convolution layer at the beginning of the block.
    - The strided convolution block has fixed kernel size of 3x3 and 1 layer of padding with zeros.
    - The number of channels is adjusted at the beginning and end of the block through 1x1 convolutional layers.
    - The number of internal channels is by default the same as the number of output channels, but
      min_inner_channels can override the behaviour.
    """

    def __init__(
        self,
        mode: Literal["top-down", "bottom-up"],
        c_in: int,
        c_out: int,
        conv_strides: tuple[int],
        min_inner_channels: Union[int, None] = None,
        nonlin: Callable = nn.LeakyReLU(),
        resample: bool = False,
        res_block_kernel: Optional[Union[int, Iterable[int]]] = None,
        groups: int = 1,
        batchnorm: bool = True,
        res_block_type: Union[str, None] = None,
        dropout: Union[float, None] = None,
        gated: Union[bool, None] = None,
        conv2d_bias: bool = True,
        # lowres_input: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        mode: Literal["top-down", "bottom-up"]
            The type of resampling performed in the initial strided convolution of the block.
            If "bottom-up" downsampling of a factor 2 is done.
            If "top-down" upsampling of a factor 2 is done.
        c_in: int
            The number of input channels.
        c_out: int
            The number of output channels.
        min_inner_channels: int, optional
            The number of channels used in the inner layer of this module.
            Default is `None`, meaning that the number of inner channels is set to `c_out`.
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `nn.LeakyReLU`.
        resample: bool, optional
            Whether to perform resampling in the first convolutional layer.
            If `False`, the first convolutional layer just maps the input to a tensor with
            `inner_channels` channels through 1x1 convolution. Default is `False`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        groups: int, optional
            The number of groups to consider in the convolutions. Default is 1.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        gated: bool, optional
            Whether to use gated layer. Default is `None`.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        """
        super().__init__()
        assert mode in ["top-down", "bottom-up"]

        conv_layer: ConvType = getattr(nn, f"Conv{len(conv_strides)}d")
        transp_conv_layer: ConvType = getattr(nn, f"ConvTranspose{len(conv_strides)}d")

        if min_inner_channels is None:
            min_inner_channels = 0
        # inner_channels is the number of channels used in the inner layers
        # of ResBlockWithResampling
        inner_channels = max(c_out, min_inner_channels)

        # Define first conv layer to change num channels and/or up/downsample
        if resample:
            if mode == "bottom-up":  # downsample
                self.pre_conv = conv_layer(
                    in_channels=c_in,
                    out_channels=inner_channels,
                    kernel_size=3,
                    padding=1,
                    stride=conv_strides,
                    groups=groups,
                    bias=conv2d_bias,
                )
            elif mode == "top-down":  # upsample
                self.pre_conv = transp_conv_layer(
                    in_channels=c_in,
                    kernel_size=3,
                    out_channels=inner_channels,
                    padding=1,  # TODO maybe don't hardcode this?
                    stride=conv_strides,
                    groups=groups,
                    output_padding=1 if len(conv_strides) == 2 else (0, 1, 1),
                    bias=conv2d_bias,
                )
        elif c_in != inner_channels:
            self.pre_conv = conv_layer(
                c_in, inner_channels, 1, groups=groups, bias=conv2d_bias
            )
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_channels,
            conv_strides=conv_strides,
            nonlin=nonlin,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            dropout=dropout,
            gated=gated,
            block_type=res_block_type,
            conv2d_bias=conv2d_bias,
        )

        # Define last conv layer to get correct num output channels
        if inner_channels != c_out:
            self.post_conv = conv_layer(
                inner_channels, c_out, 1, groups=groups, bias=conv2d_bias
            )
        else:
            self.post_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input # TODO add shape

        Returns
        -------
        torch.Tensor
            output # TODO add shape
        """
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        x = self.res(x)

        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownDeterministicResBlock(ResBlockWithResampling):
    """Resnet block for top-down deterministic layers."""

    def __init__(self, *args, upsample: bool = False, **kwargs):
        kwargs["resample"] = upsample
        super().__init__("top-down", *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):
    """Resnet block for bottom-up deterministic layers."""

    def __init__(self, *args, downsample: bool = False, **kwargs):
        kwargs["resample"] = downsample
        super().__init__("bottom-up", *args, **kwargs)


class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer.

    It consists of one or a stack of `BottomUpDeterministicResBlock`'s.
    The outputs are the so-called `bu_values` that are later used in the Decoder to update the
    generative distributions.

    NOTE: When Lateral Contextualization is Enabled (i.e., `enable_multiscale=True`),
    the low-res lateral input is first fed through a BottomUpDeterministicBlock (BUDB)
    (without downsampling), and then merged to the latent tensor produced by the primary flow
    of the `BottomUpLayer` through the `MergeLowRes` layer. It is meaningful to remark that
    the BUDB that takes care of encoding the low-res input can be either shared with the
    primary flow (and in that case it is the "same_size" BUDB (or stack of BUDBs) -> see `self.net`),
    or can be a deep-copy of the primary flow's BUDB.
    This behaviour is controlled by `lowres_separate_branch` parameter.
    """

    def __init__(
        self,
        n_res_blocks: int,
        n_filters: int,
        conv_strides: tuple[int] = (2, 2),
        downsampling_steps: int = 0,
        nonlin: Optional[Callable] = None,
        batchnorm: bool = True,
        dropout: Optional[float] = None,
        res_block_type: Optional[str] = None,
        res_block_kernel: Optional[int] = None,
        gated: Optional[bool] = None,
        enable_multiscale: bool = False,
        multiscale_lowres_size_factor: Optional[int] = None,
        lowres_separate_branch: bool = False,
        multiscale_retain_spatial_dims: bool = False,
        decoder_retain_spatial_dims: bool = False,
        output_expected_shape: Optional[Iterable[int]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_res_blocks: int
            Number of `BottomUpDeterministicResBlock` modules stacked in this layer.
        n_filters: int
            Number of channels present through out the layers of this block.
        downsampling_steps: int, optional
            Number of downsampling steps that has to be done in this layer (typically 1).
            Default is 0.
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `None`.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        gated: bool, optional
            Whether to use gated layer. Default is `None`.
        enable_multiscale: bool, optional
            Whether to enable multiscale (Lateral Contextualization) or not. Default is `False`.
        multiscale_lowres_size_factor: int, optional
            A factor the expresses the relative size of the primary flow tensor with respect to the
            lower-resolution lateral input tensor. Default in `None`.
        lowres_separate_branch: bool, optional
            Whether the residual block(s) encoding the low-res input should be shared (`False`) or
            not (`True`) with the primary flow "same-size" residual block(s). Default is `False`.
        multiscale_retain_spatial_dims: bool, optional
            Whether to pad the latent tensor resulting from the bottom-up layer's primary flow
            to match the size of the low-res input. Default is `False`.
        decoder_retain_spatial_dims: bool, optional
            Whether in the corresponding top-down layer the shape of tensor is retained between
            input and output. Default is `False`.
        output_expected_shape: Iterable[int], optional
            The expected shape of the layer output (only used if `enable_multiscale == True`).
            Default is `None`.
        """
        super().__init__()

        # Define attributes for Lateral Contextualization
        self.enable_multiscale = enable_multiscale
        self.lowres_separate_branch = lowres_separate_branch
        self.multiscale_retain_spatial_dims = multiscale_retain_spatial_dims
        self.multiscale_lowres_size_factor = multiscale_lowres_size_factor
        self.decoder_retain_spatial_dims = decoder_retain_spatial_dims
        self.output_expected_shape = output_expected_shape
        assert self.output_expected_shape is None or self.enable_multiscale is True

        bu_blocks_downsized = []
        bu_blocks_samesize = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1
            block = BottomUpDeterministicResBlock(
                conv_strides=conv_strides,
                c_in=n_filters,
                c_out=n_filters,
                nonlin=nonlin,
                downsample=do_resample,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                gated=gated,
            )
            if do_resample:
                bu_blocks_downsized.append(block)
            else:
                bu_blocks_samesize.append(block)

        self.net_downsized = nn.Sequential(*bu_blocks_downsized)
        self.net = nn.Sequential(*bu_blocks_samesize)

        # Using the same net for the low resolution (and larger sized image)
        self.lowres_net = self.lowres_merge = None
        if self.enable_multiscale:
            self._init_multiscale(
                n_filters=n_filters,
                conv_strides=conv_strides,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
            )

        # msg = f'[{self.__class__.__name__}] McEnabled:{int(enable_multiscale)} '
        # if enable_multiscale:
        #     msg += f'McParallelBeam:{int(multiscale_retain_spatial_dims)} McFactor{multiscale_lowres_size_factor}'
        # print(msg)

    def _init_multiscale(
        self,
        nonlin: Callable = None,
        n_filters: int = None,
        conv_strides: tuple[int] = (2, 2),
        batchnorm: bool = None,
        dropout: float = None,
        res_block_type: str = None,
    ) -> None:
        """
        Bottom-up layer's method that initializes the LC modules.

        Defines the modules responsible of merging compressed lateral inputs to the
        outputs of the primary flow at different hierarchical levels in the
        multiresolution approach (LC). Specifically, the method initializes `lowres_net`
        , which is a stack of `BottomUpDeterministicBlock`'s (w/out downsampling) that
        takes care of additionally processing the low-res input, and `lowres_merge`,
        which is the module responsible of merging the compressed lateral input to the
        main flow.

        NOTE: The merge modality is set by default to "residual", meaning that the
        merge layer performs concatenation on dim=1, followed by 1x1 convolution and
        a Residual Gated block.

        Parameters
        ----------
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `None`.
        n_filters: int
            Number of channels present through out the layers of this block.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        """
        self.lowres_net = self.net
        if self.lowres_separate_branch:
            self.lowres_net = deepcopy(self.net)

        self.lowres_merge = MergeLowRes(
            channels=n_filters,
            conv_strides=conv_strides,
            merge_type="residual",
            nonlin=nonlin,
            batchnorm=batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
            multiscale_lowres_size_factor=self.multiscale_lowres_size_factor,
        )

    def forward(
        self, x: torch.Tensor, lowres_x: Union[torch.Tensor, None] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input of the `BottomUpLayer`, i.e., the input image or the output of the
            previous layer.
        lowres_x: torch.Tensor, optional
            The low-res input used for Lateral Contextualization (LC). Default is `None`.

        NOTE: first returned tensor is used as input for the next BU layer, while the second
        tensor is the bu_value passed to the top-down layer.
        """
        # The input is fed through the residual downsampling block(s)
        primary_flow = self.net_downsized(x)
        # The downsampling output is fed through additional residual block(s)
        primary_flow = self.net(primary_flow)

        # If LC is not used, simply return output of primary-flow
        if self.enable_multiscale is False:
            assert lowres_x is None
            return primary_flow, primary_flow

        if lowres_x is not None:
            # First encode the low-res lateral input
            lowres_flow = self.lowres_net(lowres_x)
            # Then pass the result through the MergeLowRes layer
            merged = self.lowres_merge(primary_flow, lowres_flow)
        else:
            merged = primary_flow

        # NOTE: Explanation of possible cases for the conditionals:
        # - if both are `True` -> `merged` has the same spatial dims as the input (`x`) since
        #   spatial dims are retained by padding `primary_flow` in `MergeLowRes`. This is
        #   OK for the corresp TopDown layer, as it also retains spatial dims.
        # - if both are `False` -> `merged`'s spatial dims are equal to `self.net_downsized(x)`,
        #   since no padding is done in `MergeLowRes` and, instead, the lowres input is cropped.
        #   This is OK for the corresp TopDown layer, as it also halves the spatial dims.
        # - if 1st is `False` and 2nd is `True` -> not a concern, it cannot happen
        #   (see lvae.py, line 111, intialization of `multiscale_decoder_retain_spatial_dims`).
        if (
            self.multiscale_retain_spatial_dims is False
            or self.decoder_retain_spatial_dims is True
        ):
            return merged, merged

        # NOTE: if we reach here, it means that `multiscale_retain_spatial_dims` is `True`,
        # but `decoder_retain_spatial_dims` is `False`, meaning that merging LC preserves
        # the spatial dimensions, but at the same time we don't want to retain the spatial
        # dims in the corresponding top-down layer. Therefore, we need to crop the tensor.
        if self.output_expected_shape is not None:
            expected_shape = self.output_expected_shape
        else:
            fac = self.multiscale_lowres_size_factor
            expected_shape = (merged.shape[-2] // fac, merged.shape[-1] // fac)
            assert merged.shape[-2:] != expected_shape

        # Crop the resulting tensor so that it matches with the Decoder
        value_to_use_in_topdown = crop_img_tensor(merged, expected_shape)
        return merged, value_to_use_in_topdown


class MergeLayer(nn.Module):
    """
    Layer class that merges two or more input tensors.

    Merges two or more (B, C, [Z], Y, X) input tensors by concatenating
    them along dim=1 and passes the result through:
    a) a convolutional 1x1 layer (`merge_type == "linear"`), or
    b) a convolutional 1x1 layer and then a gated residual block (`merge_type == "residual"`), or
    c) a convolutional 1x1 layer and then an ungated residual block (`merge_type == "residual_ungated"`).
    """

    def __init__(
        self,
        merge_type: Literal["linear", "residual", "residual_ungated"],
        channels: Union[int, Iterable[int]],
        conv_strides: tuple[int] = (2, 2),
        nonlin: Callable = nn.LeakyReLU(),
        batchnorm: bool = True,
        dropout: Optional[float] = None,
        res_block_type: Optional[str] = None,
        res_block_kernel: Optional[int] = None,
        conv2d_bias: Optional[bool] = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        merge_type: Literal["linear", "residual", "residual_ungated"]
            The type of merge done in the layer. It can be chosen between "linear",
            "residual", and "residual_ungated". Check the class docstring for more
            information about the behaviour of different merge modalities.
        channels: Union[int, Iterable[int]]
            The number of channels used in the convolutional blocks of this layer.
            If it is an `int`:
                - 1st 1x1 Conv2d: in_channels=2*channels, out_channels=channels
                - (Optional) ResBlock: in_channels=channels, out_channels=channels
            If it is an Iterable (must have `len(channels)==3`):
                - 1st 1x1 Conv2d: in_channels=sum(channels[:-1]),
                out_channels=channels[-1]
                - (Optional) ResBlock: in_channels=channels[-1],
                out_channels=channels[-1]
        conv_strides: tuple, optional
            The strides used in the convolutions. Default is `(2, 2)`.
        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `nn.LeakyReLU`.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
            Default is `None`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared
            kernel.
            Default is `None`.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        """
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3

        self.conv_layer: ConvType = getattr(nn, f"Conv{len(conv_strides)}d")

        if merge_type == "linear":
            self.layer = self.conv_layer(
                sum(channels[:-1]), channels[-1], 1, bias=conv2d_bias
            )
        elif merge_type == "residual":
            self.layer = nn.Sequential(
                self.conv_layer(
                    sum(channels[:-1]), channels[-1], 1, padding=0, bias=conv2d_bias
                ),
                ResidualGatedBlock(
                    conv_strides=conv_strides,
                    channels=channels[-1],
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    block_type=res_block_type,
                    kernel=res_block_kernel,
                    conv2d_bias=conv2d_bias,
                ),
            )
        elif merge_type == "residual_ungated":
            self.layer = nn.Sequential(
                self.conv_layer(
                    sum(channels[:-1]), channels[-1], 1, padding=0, bias=conv2d_bias
                ),
                ResidualBlock(
                    conv_strides=conv_strides,
                    channels=channels[-1],
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    block_type=res_block_type,
                    kernel=res_block_kernel,
                    conv2d_bias=conv2d_bias,
                ),
            )

    def forward(self, *args) -> torch.Tensor:

        # Concatenate the input tensors along dim=1
        x = torch.cat(args, dim=1)

        # Pass the concatenated tensor through the conv layer
        x = self.layer(x)

        return x


class MergeLowRes(MergeLayer):
    """
    Child class of `MergeLayer`.

    Specifically designed to merge the low-resolution patches
    that are used in Lateral Contextualization approach.
    """

    def __init__(self, *args, **kwargs):
        self.retain_spatial_dims = kwargs.pop("multiscale_retain_spatial_dims")
        self.multiscale_lowres_size_factor = kwargs.pop("multiscale_lowres_size_factor")
        super().__init__(*args, **kwargs)

    def forward(self, latent: torch.Tensor, lowres: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        latent: torch.Tensor
            The output latent tensor from previous layer in the LVAE hierarchy.
        lowres: torch.Tensor
            The low-res patch image to be merged to increase the context.
        """
        # TODO: treat (X, Y) and Z differently (e.g., line 762)
        if self.retain_spatial_dims:
            # Pad latent tensor to match lowres tensor's shape
            # Output.shape == Lowres.shape (== Input.shape),
            # where Input is the input to the BU layer
            latent = pad_img_tensor(latent, lowres.shape[2:])
        else:
            # Crop lowres tensor to match latent tensor's shape
            lz, ly, lx = lowres.shape[2:]
            z = lz // self.multiscale_lowres_size_factor
            y = ly // self.multiscale_lowres_size_factor
            x = lx // self.multiscale_lowres_size_factor
            z_pad = (lz - z) // 2
            y_pad = (ly - y) // 2
            x_pad = (lx - x) // 2
            lowres = lowres[:, :, z_pad:-z_pad, y_pad:-y_pad, x_pad:-x_pad]

        return super().forward(latent, lowres)


class SkipConnectionMerger(MergeLayer):
    """Specialized `MergeLayer` module, handles skip connections in the model."""

    def __init__(
        self,
        nonlin: Callable,
        channels: Union[int, Iterable[int]],
        batchnorm: bool,
        dropout: float,
        res_block_type: str,
        conv_strides: tuple[int] = (2, 2),
        merge_type: Literal["linear", "residual", "residual_ungated"] = "residual",
        conv2d_bias: bool = True,
        res_block_kernel: Optional[int] = None,
    ):
        """
        Constructor.

        nonlin: Callable, optional
            The non-linearity function used in the block. Default is `nn.LeakyReLU`.
        channels: Union[int, Iterable[int]]
            The number of channels used in the convolutional blocks of this layer.
            If it is an `int`:
                - 1st 1x1 Conv2d: in_channels=2*channels, out_channels=channels
                - (Optional) ResBlock: in_channels=channels, out_channels=channels
            If it is an Iterable (must have `len(channels)==3`):
                - 1st 1x1 Conv2d: in_channels=sum(channels[:-1]), out_channels=channels[-1]
                - (Optional) ResBlock: in_channels=channels[-1], out_channels=channels[-1]
        batchnorm: bool
            Whether to use batchnorm layers.
        dropout: float
            The dropout probability in dropout layers. If `None` dropout is not used.
        res_block_type: str
            A string specifying the structure of residual block.
            Check `ResidualBlock` doscstring for more information.
        conv_strides: tuple, optional
            The strides used in the convolutions. Default is `(2, 2)`.
        merge_type: Literal["linear", "residual", "residual_ungated"]
            The type of merge done in the layer. It can be chosen between "linear", "residual", and "residual_ungated".
            Check the class docstring for more information about the behaviour of different merge modalities.
        conv2d_bias: bool, optional
            Whether to use bias term in convolutions. Default is `True`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        """
        super().__init__(
            conv_strides=conv_strides,
            channels=channels,
            nonlin=nonlin,
            merge_type=merge_type,
            batchnorm=batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            res_block_kernel=res_block_kernel,
            conv2d_bias=conv2d_bias,
        )


class TopDownLayer(nn.Module):
    """Top-down inference layer.

    It includes:
        - Stochastic sampling,
        - Computation of KL divergence,
        - A small deterministic ResNet that performs upsampling.

    NOTE 1:
        The algorithm for generative inference approximately works as follows:
            - p_params = output of top-down layer above
            - bu = inferred bottom-up value at this layer
            - q_params = merge(bu, p_params)
            - z = stochastic_layer(q_params)
            - (optional) get and merge skip connection from prev top-down layer
            - top-down deterministic ResNet

    NOTE 2:
        The Top-Down layer can work in two modes: inference and prediction/generative.
        Depending on the particular mode, it follows distinct behaviours:
        - In inference mode, parameters of q(z_i|z_i+1) are obtained from the inference path,
        by merging outcomes of bottom-up and top-down passes. The exception is the top layer,
        in which the parameters of q(z_L|x) are set as the output of the topmost bottom-up layer.
        - On the contrary in predicition/generative mode, parameters of q(z_i|z_i+1) can be obtained
        once again by merging bottom-up and top-down outputs (CONDITIONAL GENERATION), or it is
        possible to directly sample from the prior p(z_i|z_i+1) (UNCONDITIONAL GENERATION).

    NOTE 3:
        When doing unconditional generation, bu_value is not available. Hence the
        merge layer is not used, and z is sampled directly from p_params.

    NOTE 4:
        If this is the top layer, at inference time, the uppermost bottom-up value
        is used directly as q_params, and p_params are defined in this layer
        (while they are usually taken from the previous layer), and can be learned.
    """

    def __init__(
        self,
        z_dim: int,
        n_res_blocks: int,
        n_filters: int,
        conv_strides: tuple[int],
        is_top_layer: bool = False,
        upsampling_steps: Union[int, None] = None,
        nonlin: Union[Callable, None] = None,
        merge_type: Union[
            Literal["linear", "residual", "residual_ungated"], None
        ] = None,
        batchnorm: bool = True,
        dropout: Union[float, None] = None,
        stochastic_skip: bool = False,
        res_block_type: Union[str, None] = None,
        res_block_kernel: Union[int, None] = None,
        groups: int = 1,
        gated: Union[bool, None] = None,
        learn_top_prior: bool = False,
        top_prior_param_shape: Union[Iterable[int], None] = None,
        analytical_kl: bool = False,
        retain_spatial_dims: bool = False,
        vanilla_latent_hw: Union[Iterable[int], None] = None,
        input_image_shape: Union[tuple[int, int], None] = None,
        normalize_latent_factor: float = 1.0,
        conv2d_bias: bool = True,
        stochastic_use_naive_exponential: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        z_dim: int
            The size of the latent space.
        n_res_blocks: int
            The number of TopDownDeterministicResBlock blocks
        n_filters: int
            The number of channels present through out the layers of this block.
        conv_strides: tuple, optional
            The strides used in the convolutions. Default is `(2, 2)`.
        is_top_layer: bool, optional
            Whether the current layer is at the top of the Decoder hierarchy. Default is `False`.
        upsampling_steps: int, optional
            The number of upsampling steps that has to be done in this layer (typically 1).
            Default is `None`.
        nonlin: Callable, optional
            The non-linearity function used in the block (e.g., `nn.ReLU`). Default is `None`.
        merge_type: Literal["linear", "residual", "residual_ungated"], optional
            The type of merge done in the layer. It can be chosen between "linear", "residual",
            and "residual_ungated". Check the `MergeLayer` class docstring for more information
            about the behaviour of different merging modalities. Default is `None`.
        batchnorm: bool, optional
            Whether to use batchnorm layers. Default is `True`.
        dropout: float, optional
            The dropout probability in dropout layers. If `None` dropout is not used.
            Default is `None`.
        stochastic_skip: bool, optional
            Whether to use skip connections between previous top-down layer's output and this layer's stochastic output.
            Stochastic skip connection allows the previous layer's output has a way to directly reach this hierarchical
            level, hence facilitating the gradient flow during backpropagation. Default is `False`.
        res_block_type: str, optional
            A string specifying the structure of residual block.
            Check `ResidualBlock` documentation for more information.
            Default is `None`.
        res_block_kernel: Union[int, Iterable[int]], optional
            The kernel size used in the convolutions of the residual block.
            It can be either a single integer or a pair of integers defining the squared kernel.
            Default is `None`.
        groups: int, optional
            The number of groups to consider in the convolutions. Default is 1.
        gated: bool, optional
            Whether to use gated layer in `ResidualBlock`. Default is `None`.
        learn_top_prior:
            Whether to set the top prior as learnable.
            If this is set to `False`, in the top-most layer the prior will be N(0,1).
            Otherwise, we will still have a normal distribution whose parameters will be learnt.
            Default is `False`.
        top_prior_param_shape: Iterable[int], optional
            The size of the tensor which expresses the mean and the variance
            of the prior for the top most layer. Default is `None`.
        analytical_kl: bool, optional
            If True, KL divergence is calculated according to the analytical formula.
            Otherwise, an MC approximation using sampled latents is calculated.
            Default is `False`.
        retain_spatial_dims: bool, optional
            If `True`, the size of Encoder's latent space is kept to `input_image_shape` within the topdown layer.
            This implies that the oput spatial size equals the input spatial size.
            To achieve this, we centercrop the intermediate representation.
            Default is `False`.
        vanilla_latent_hw: Iterable[int], optional
            The shape of the latent tensor used for prediction (i.e., it influences the computation of restricted KL).
            Default is `None`.
        input_image_shape: Tuple[int, int], optionalut
            The shape of the input image tensor.
            When `retain_spatial_dims` is set to `True`, this is used to ensure that the shape of this layer
            output has the same shape as the input. Default is `None`.
        normalize_latent_factor: float, optional
            A factor used to normalize the latent tensors `q_params`.
            Specifically, normalization is done by dividing the latent tensor by this factor.
            Default is 1.0.
        conv2d_bias: bool, optional
            Whether to use bias term is the convolutional blocks of this layer.
            Default is `True`.
        stochastic_use_naive_exponential: bool, optional
            If `False`, in the NormalStochasticBlock2d exponentials are computed according
            to the alternative definition provided by `StableExponential` class.
            This should improve numerical stability in the training process.
            Default is `False`.
        """
        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip
        self.learn_top_prior = learn_top_prior
        self.analytical_kl = analytical_kl
        self.retain_spatial_dims = retain_spatial_dims
        self.input_image_shape = (
            input_image_shape if len(conv_strides) == 3 else input_image_shape[1:]
        )
        self.latent_shape = self.input_image_shape if self.retain_spatial_dims else None
        self.normalize_latent_factor = normalize_latent_factor
        self._vanilla_latent_hw = vanilla_latent_hw  # TODO: check this, it is not used

        # Define top layer prior parameters, possibly learnable
        if is_top_layer:
            self.top_prior_params = nn.Parameter(
                torch.zeros(top_prior_param_shape), requires_grad=learn_top_prior
            )

        # Upsampling steps left to do in this layer
        ups_left = upsampling_steps

        # Define deterministic top-down block, which is a sequence of deterministic
        # residual blocks with (optional) upsampling.
        block_list = []
        for _ in range(n_res_blocks):
            do_resample = False
            if ups_left > 0:
                do_resample = True
                ups_left -= 1
            block_list.append(
                TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    conv_strides=conv_strides,
                    nonlin=nonlin,
                    upsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    res_block_kernel=res_block_kernel,
                    gated=gated,
                    conv2d_bias=conv2d_bias,
                    groups=groups,
                )
            )
        self.deterministic_block = nn.Sequential(*block_list)

        # Define stochastic block with convolutions

        self.stochastic = NormalStochasticBlock(
            c_in=n_filters,
            c_vars=z_dim,
            c_out=n_filters,
            conv_dims=len(conv_strides),
            transform_p_params=(not is_top_layer),
            vanilla_latent_hw=vanilla_latent_hw,
            use_naive_exponential=stochastic_use_naive_exponential,
        )

        if not is_top_layer:
            # Merge layer: it combines bottom-up inference and top-down
            # generative outcomes to give posterior parameters
            self.merge = MergeLayer(
                channels=n_filters,
                conv_strides=conv_strides,
                merge_type=merge_type,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                conv2d_bias=conv2d_bias,
            )

            # Skip connection that goes around the stochastic top-down layer
            if stochastic_skip:
                self.skip_connection_merger = SkipConnectionMerger(
                    channels=n_filters,
                    conv_strides=conv_strides,
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    merge_type=merge_type,
                    conv2d_bias=conv2d_bias,
                    res_block_kernel=res_block_kernel,
                )

    def sample_from_q(
        self,
        input_: torch.Tensor,
        bu_value: torch.Tensor,
        var_clip_max: Optional[float] = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Method computes the latent inference distribution q(z_i|z_{i+1}).

        Used for sampling a latent tensor from it.

        Parameters
        ----------
        input_: torch.Tensor
            The input tensor to the layer, which is the output of the top-down layer.
        bu_value: torch.Tensor
            The tensor defining the parameters /mu_q and /sigma_q computed during the
            bottom-up deterministic pass at the correspondent hierarchical layer.
        var_clip_max: float, optional
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped. Default is `None`.
        mask: Union[None, torch.Tensor], optional
            A tensor that is used to mask the sampled latent tensor. Default is `None`.
        """
        if self.is_top_layer:  # In top layer, we don't merge bu_value with p_params
            q_params = bu_value
        else:
            # NOTE: Here the assumption is that the vampprior is only applied on the top layer.
            n_img_prior = None
            p_params = self.get_p_params(input_, n_img_prior)
            q_params = self.merge(bu_value, p_params)

        sample = self.stochastic.sample_from_q(q_params, var_clip_max)

        if mask:
            return sample[mask]

        return sample

    def get_p_params(
        self,
        input_: torch.Tensor,
        n_img_prior: int,
    ) -> torch.Tensor:
        """Return the parameters of the prior distribution p(z_i|z_{i+1}).

        The parameters depend on the hierarchical level of the layer:
        - if it is the topmost level, parameters are the ones of the prior.
        - else, the input from the layer above is the parameters itself.

        Parameters
        ----------
        input_: torch.Tensor
            The input tensor to the layer, which is the output of the top-down layer above.
        n_img_prior: int
            The number of images to be generated from the unconditional prior distribution p(z_L).
        """
        p_params = None

        # If top layer, define p_params as the ones of the prior p(z_L)
        if self.is_top_layer:
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.expand(n_img_prior, -1, -1, -1)

        # Else the input from the layer above is p_params itself
        else:
            p_params = input_

        return p_params

    def forward(
        self,
        input_: Union[torch.Tensor, None] = None,
        skip_connection_input: Union[torch.Tensor, None] = None,
        inference_mode: bool = False,
        bu_value: Union[torch.Tensor, None] = None,
        n_img_prior: Union[int, None] = None,
        forced_latent: Union[torch.Tensor, None] = None,
        force_constant_output: bool = False,
        mode_pred: bool = False,
        use_uncond_mode: bool = False,
        var_clip_max: Union[float, None] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        input_: torch.Tensor, optional
            The input tensor to the layer, which is the output of the top-down layer.
            Default is `None`.
        skip_connection_input: torch.Tensor, optional
            The tensor brought by the skip connection between the current and the
            previous top-down layer.
            Default is `None`.
        inference_mode: bool, optional
            Whether the layer is in inference mode. See NOTE 2 in class description
            for more info.
            Default is `False`.
        bu_value: torch.Tensor, optional
            The tensor defining the parameters /mu_q and /sigma_q computed during the
            bottom-up deterministic pass
            at the correspondent hierarchical layer. Default is `None`.
        n_img_prior: int, optional
            The number of images to be generated from the unconditional prior
            distribution p(z_L).
            Default is `None`.
        forced_latent: torch.Tensor, optional
            A pre-defined latent tensor. If it is not `None`, than it is used as the
            actual latent tensor and,
            hence, sampling does not happen. Default is `None`.
        force_constant_output: bool, optional
            Whether to copy the first sample (and rel. distrib parameters) over the
            whole batch.
            This is used when doing experiment from the prior - q is not used.
            Default is `False`.
        mode_pred: bool, optional
            Whether the model is in prediction mode. Default is `False`.
        use_uncond_mode: bool, optional
            Whether to use the uncoditional distribution p(z) to sample latents in
            prediction mode.
        var_clip_max: float
            The maximum value reachable by the log-variance of the latent distribution.
            Values exceeding this threshold are clipped.
        """
        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        p_params = self.get_p_params(input_, n_img_prior)

        # Get the parameters for the latent distribution to sample from
        if inference_mode:  # TODO What's this ? reuse Fede's code?
            if self.is_top_layer:
                q_params = bu_value
                if mode_pred is False:
                    assert p_params.shape[2:] == bu_value.shape[2:], (
                        "Spatial dimensions of p_params and bu_value should match. "
                        f"Instead, we got p_params={p_params.shape[2:]} and "
                        f"bu_value={bu_value.shape[2:]}."
                    )
            else:
                if use_uncond_mode:
                    q_params = p_params
                else:
                    assert p_params.shape[2:] == bu_value.shape[2:], (
                        "Spatial dimensions of p_params and bu_value should match. "
                        f"Instead, we got p_params={p_params.shape[2:]} and "
                        f"bu_value={bu_value.shape[2:]}."
                    )
                    q_params = self.merge(bu_value, p_params)
        else:  # generative mode, q is not used, we sample from p(z_i | z_{i+1})
            q_params = None

        # NOTE: Sampling is done either from q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on the mode (hence, in practice, by checking whether q_params is None).

        # Normalization of latent space parameters for stablity.
        # See Very deep VAEs generalize autoregressive models.
        if self.normalize_latent_factor:
            q_params = q_params / self.normalize_latent_factor

        # Sample (and process) a latent tensor in the stochastic layer
        x, data_stoch = self.stochastic(
            p_params=p_params,
            q_params=q_params,
            forced_latent=forced_latent,
            force_constant_output=force_constant_output,
            analytical_kl=self.analytical_kl,
            mode_pred=mode_pred,
            use_uncond_mode=use_uncond_mode,
            var_clip_max=var_clip_max,
        )
        # Merge skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            x = self.skip_connection_merger(x, skip_connection_input)
        if self.retain_spatial_dims:
            # NOTE: we assume that one topdown layer will have exactly one upscaling layer.

            # NOTE: in case, in the Bottom-Up layer, LC retains spatial dimensions,
            # we have the following (see `MergeLowRes`):
            # - the "primary-flow" tensor is padded to match the low-res patch size
            #   (e.g., from 32x32 to 64x64)
            # - padded tensor is then merged with the low-res patch (concatenation
            #   along dim=1 + convolution)
            # Therefore, we need to do the symmetric operation here, that is to
            # crop `x` for the same amount we padded it in the correspondent BU layer.

            # NOTE: cropping is done to retain the shape of the input in the output.
            # Therefore we need it only in the case `x` is the same shape of the input,
            # because that's the only case in which we need to retain the shape.
            # Here, it must be strictly greater than half the input shape, which is
            # the case if and only if `x.shape == self.latent_shape`.
            rescale = (
                np.array((1, 2, 2)) if len(self.latent_shape) == 3 else np.array((2, 2))
            )  # TODO better way?
            new_latent_shape = tuple(np.array(self.latent_shape) // rescale)
            if x.shape[-1] > new_latent_shape[-1]:
                x = crop_img_tensor(x, new_latent_shape)
        # TODO: `retain_spatial_dims` is the same for all the TD layers.
        # How to handle the case in which we do not have LC for all layers?
        # The answer is in `self.latent_shape`, which is equal to `input_image_shape`
        # (e.g., (64, 64)) if `retain_spatial_dims` is `True`, else it is `None`.
        # Last top-down block (sequence of residual blocks w\ upsampling)
        x = self.deterministic_block(x)
        # Save some metrics that will be used in the loss computation
        keys = [
            "z",
            "kl_samplewise",
            "kl_samplewise_restricted",
            "kl_spatial",
            "kl_channelwise",
            "logprob_q",
            "qvar_max",
        ]
        data = {k: data_stoch.get(k, None) for k in keys}
        data["q_mu"] = None
        data["q_lv"] = None
        if data_stoch["q_params"] is not None:
            q_mu, q_lv = data_stoch["q_params"]
            data["q_mu"] = q_mu
            data["q_lv"] = q_lv
        return x, data

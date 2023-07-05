import sys
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple

# TODO add docstings, typing


def conv3x3(
    conv_func, in_channels, out_channels, stride=1, padding=1, bias=True, groups=1
):
    return conv_func(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


def upconv2x2(
    conv_dim: int, conv_func_transpose, in_channels, out_channels, mode="transpose"
):
    # TODO add better conv definition
    if mode == "transpose":
        return conv_func_transpose(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv1x1(conv_dim, in_channels, out_channels),
        )


def conv1x1(conv_dim: int, in_channels, out_channels, groups=1):
    conv_func = getattr(nn, f"Conv{conv_dim}d")
    return conv_func(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, conv_dim, in_channels, out_channels, pooling=True, n2v2=False):
        super().__init__()

        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.n2v2 = n2v2
        self.conv_func = getattr(nn, f"Conv{self.conv_dim}d")

        self.conv1 = conv3x3(self.conv_func, self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.conv_func, self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = (
                getattr(sys.__name__, f"BlurPool{2}d")(
                    channels=1, filt_size=2, stride=2
                )
                if self.n2v2
                else getattr(nn, f"MaxPool{self.conv_dim}d")(kernel_size=2, stride=2)
            )
        self.bn = getattr(nn, f"BatchNorm{self.conv_dim}d")(self.out_channels)

    def forward(self, x):
        x = functional.relu(self.bn(self.conv1(x)))
        x = functional.relu(self.bn(self.conv2(x)))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        conv_dim,
        in_channels,
        out_channels,
        merge_mode="concat",
        up_mode="transpose",
        skip=False,
    ):
        super().__init__()

        self.conv_dim = conv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.skip = skip
        self.conv_func = getattr(nn, f"Conv{self.conv_dim}d")
        self.conv_func_transpose = getattr(nn, f"ConvTranspose{self.conv_dim}d")

        self.upconv = upconv2x2(
            self.conv_dim,
            self.conv_func_transpose,
            self.in_channels,
            self.out_channels,
            mode=self.up_mode,
        )

        if self.merge_mode == "concat":
            self.conv1 = conv3x3(
                self.conv_func, 2 * self.out_channels, self.out_channels
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.conv_func, self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.conv_func, self.out_channels, self.out_channels)
        self.bn = getattr(nn, f"BatchNorm{self.conv_dim}d")(self.out_channels)

    def forward(self, from_down, from_up, skip=False):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway.
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        if self.skip:
            x = from_down
        x = functional.relu(self.bn(self.conv1(x)))
        x = functional.relu(self.bn(self.conv2(x)))
        return x


class Conv_Block_tf(nn.Module):
    def __init__(
        self,
        conv_dim,
        in_channels,
        out_channels,
        stride=1,
        padding=1,
        bias=True,
        groups=1,
        activation="ReLU",
        dropout_perc=0,
        use_batch_norm=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.conv = getattr(nn, f"Conv{conv_dim}d")(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        self.batch_norm = getattr(nn, f"BatchNorm{conv_dim}d")(out_channels)
        self.dropout = (
            getattr(nn, f"Dropout{conv_dim}d")(dropout_perc)
            if dropout_perc > 0
            else None
        )
        self.activation = (
            getattr(nn, f"{activation}")() if activation is not None else nn.Identity()
        )

    def forward(self, x):
        if self.use_batch_norm:
            x = self.conv(x)
            x = self.batch_norm(x)
            x = self.activation(x)
        else:
            x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def default_2d_filter():
    return (
        torch.tensor(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]
        )
        * 1
        / 16.0
    )


def default_3d_filter():
    return (
        torch.tensor(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
            [
                [2, 4, 2],
                [4, 8, 4],
                [2, 4, 2],
            ],
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
        )
        * 1
        / 64.0
    )


def padding_filter_same(filter: torch.Tensor) -> Tuple[int, ...]:
    if np.any([dim % 2 == 0 for dim in filter.shape[2:]]):
        raise ValueError("All filter dimensions must be odd")

    padded_filter = [int(torch.div(dim, 2)) for dim in filter.shape[2:]]

    return tuple(padded_filter)


def blur_operation(
    input_tensor: torch.Tensor,
    stride: Union[_size_2_t, _size_3_t] = 1,
    channels: Optional[int] = None,
    conv_mult: int = 2,
    filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Applies a spatial filter.

    Args:
    ----
        input (torch.Tensor): A 4D/5D tensor of shape NC(Z)YX.

        stride (int | tuple, optional): Stride(s) along axes. If a single value is passed, this
            value is used for both dimensions.
        conv_mult (int): Used to choose between 2D and 3D convolution. Default: 2.
        filter (torch.Tensor, optional): A 2D/3D tensor to be convolved with the input tensor
            at each spatial position, across all channels. If not provided, a default filter
    Returns:
        Blurred input
    """

    if filter is None:
        filter = getattr(sys.modules[__name__], f"default_{conv_mult}d_filter")
    # The dynamic control flow branch below does not affect the padding as only h and w are used.
    padding = padding_filter_same(filter)

    if (
        channels is not None and channels < 1
    ):  # Use Dynamic Control Flow # TODO <- what does that mean? I wish I knew :)
        _, channels, *spatial_dims = input_tensor.shape

        filter = filter.repeat((channels, [1] * len(input_tensor.shape[1:])))
        _, _, *filter_spatial_dims = filter.shape

        if torch.any(torch.tensor(spatial_dims) < torch.tensor(filter_spatial_dims)):
            return input_tensor

    # TODO: the following comment needs more clarification
    # Call functional.conv2d without using keyword arguments as that triggers a bug in fx tracing quantization.
    conv_operation = getattr(functional, f"conv{conv_mult}d")
    _ntuple = _pair if conv_mult == 2 else _triple
    return conv_operation(
        input_tensor,
        filter,
        None,
        _ntuple(stride),
        _ntuple(padding),
        _ntuple(1),
        channels,
    )


def blurmax_pool(
    input_tensor: torch.Tensor,
    kernel_size: Optional[Union[_size_2_t, _size_3_t]] = None,
    stride: Union[_size_2_t, _size_3_t] = 2,
    padding: Union[_size_2_t, _size_3_t] = 0,
    dilation: Union[_size_2_t, _size_3_t] = 1,
    ceil_mode: bool = False,
    conv_mult: int = 2,
    filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Max-pooling with anti-aliasing.
    This is a nearly drop-in replacement for PyTorch's :func:`torch.nn.functional.max_pool2d`.
    The only API difference is that the parameter ``return_indices`` is not
    available, because it is ill-defined when using anti-aliasing.
    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.
    This function can be understood as decoupling the max from the pooling, and
    inserting a low-pass filtering step between the two. Concretely, this
    function computes the max within spatial neighborhoods of shape
    ``kernel_size``, then applies an anti-aliasing filter to smooth the maxes,
    and only then pools according to ``stride``.
    See also: :func:`.blur_2d`.

    Args:
    ----
        input (torch.Tensor): A 4d tensor of shape NCHW
        kernel_size (int | tuple, optional): Size(s) of the spatial neighborhoods over which to pool.
            This is mostly commonly 2x2. If only a scalar ``s`` is provided, the
            neighborhood is of size ``(s, s)``. Default: ``(2, 2)``.
        stride (int | tuple, optional): Stride(s) along H and W axes. If a single value is passed, this
            value is used for both dimensions. Default: 2.
        padding (int | tuple, optional): implicit zero-padding to use. For the default 3x3 low-pass
            filter, ``padding=1`` (the default) returns output of the same size
            as the input. Default: 0.
        dilation (int | tuple, optional): Amount by which to "stretch" the pooling region for a given
            total size. See :class:`torch.nn.MaxPool2d`
            for our favorite explanation of how this works. Default: 1.
        ceil_mode (bool): When True, will use ceil instead of floor to compute the output shape. Default: ``False``.
        filter (torch.Tensor, optional): A 2d or 4d tensor to be cross-correlated with the input tensor
            at each spatial position, within each channel. If 4d, the structure
            is required to be ``(C, 1, kH, kW)`` where ``C`` is the number of
            channels in the input tensor and ``kH`` and ``kW`` are the spatial
            sizes of the filter.
    By default, the filter used is:
    .. code-block:: python
            [1 2 1]
            [2 4 2] * 1/16
            [1 2 1]


    Returns:
    -------
         The blurred and max-pooled input.
    """
    if kernel_size is None:
        kernel_size = (2, 2)

    pool_operation = getattr(functional, f"max_pool{conv_mult}d")
    maxs = pool_operation(
        input_tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    return blur_operation(maxs, channels=-1, stride=stride, filter=filter)


class BlurPool2d(nn.Module):
    """This module is a (nearly) drop-in replacement for :class:`torch.nn.MaxPool2d`, but with an anti-aliasing filter.
    The only API difference is that the parameter ``return_indices`` is not
    available, because it is ill-defined when using anti-aliasing.
    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.
    See :func:`.blurmax_pool2d` for details.
    """

    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html#MaxPool2d # noqa

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: torch.Tensor):
        return blurmax_pool(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            conv_mult=2,
            filter=self.filt2d,
        )


class BlurPool3d(nn.Module):
    """This module is a (nearly) drop-in replacement for :class:`torch.nn.MaxPool2d`, but with an anti-aliasing filter.
    The only API difference is that the parameter ``return_indices`` is not
    available, because it is ill-defined when using anti-aliasing.
    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.
    See :func:`.blurmax_pool2d` for details.
    """

    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html#MaxPool2d # noqa

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: torch.Tensor):
        return blurmax_pool(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            conv_mult=3,
            filter=self.filt2d,
        )

"""
Layer module.

This submodule contains layers used in the CAREamics models.
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv_Block(nn.Module):
    """
    Convolution block used in UNets.

    Convolution block consist of two convolution layers with optional batch norm,
    dropout and with a final activation function.

    The parameters are directly mapped to PyTorch Conv2D and Conv3d parameters, see
    PyTorch torch.nn.Conv2d and torch.nn.Conv3d for more information.

    Parameters
    ----------
    conv_dim : int
        Number of dimension of the convolutions, 2 or 3.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    intermediate_channel_multiplier : int, optional
        Multiplied for the number of output channels, by default 1.
    stride : int, optional
        Stride of the convolutions, by default 1.
    padding : int, optional
        Padding of the convolutions, by default 1.
    bias : bool, optional
        Bias of the convolutions, by default True.
    groups : int, optional
        Controls the connections between inputs and outputs, by default 1.
    activation : str, optional
        Activation function, by default "ReLU".
    dropout_perc : float, optional
        Dropout percentage, by default 0.
    use_batch_norm : bool, optional
        Use batch norm, by default False.
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        out_channels: int,
        intermediate_channel_multiplier: int = 1,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        groups: int = 1,
        activation: str = "ReLU",
        dropout_perc: float = 0,
        use_batch_norm: bool = False,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimension of the convolutions, 2 or 3.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        intermediate_channel_multiplier : int, optional
            Multiplied for the number of output channels, by default 1.
        stride : int, optional
            Stride of the convolutions, by default 1.
        padding : int, optional
            Padding of the convolutions, by default 1.
        bias : bool, optional
            Bias of the convolutions, by default True.
        groups : int, optional
            Controls the connections between inputs and outputs, by default 1.
        activation : str, optional
            Activation function, by default "ReLU".
        dropout_perc : float, optional
            Dropout percentage, by default 0.
        use_batch_norm : bool, optional
            Use batch norm, by default False.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = getattr(nn, f"Conv{conv_dim}d")(
            in_channels,
            out_channels * intermediate_channel_multiplier,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        self.conv2 = getattr(nn, f"Conv{conv_dim}d")(
            out_channels * intermediate_channel_multiplier,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        self.batch_norm1 = getattr(nn, f"BatchNorm{conv_dim}d")(
            out_channels * intermediate_channel_multiplier
        )
        self.batch_norm2 = getattr(nn, f"BatchNorm{conv_dim}d")(out_channels)

        self.dropout = (
            getattr(nn, f"Dropout{conv_dim}d")(dropout_perc)
            if dropout_perc > 0
            else None
        )
        self.activation = (
            getattr(nn, f"{activation}")() if activation is not None else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.use_batch_norm:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.activation(x)
        else:
            x = self.conv1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def _unpack_kernel_size(kernel_size: Union[Tuple[int], int], dim: int) -> Tuple[int]:
    """Unpack kernel_size to a tuple of ints.

    Inspired by Kornia implementation.
    """
    if isinstance(kernel_size, int):
        kernel_dims = [kernel_size for _ in range(dim)]
    else:
        kernel_dims = kernel_size
    return kernel_dims


def _compute_zero_padding(kernel_size: Union[Tuple[int], int], dim: int) -> Tuple[int]:
    """Utility function that computes zero padding tuple."""
    kernel_dims = _unpack_kernel_size(kernel_size, dim)
    return [(kd - 1) // 2 for kd in kernel_dims]


def get_pascal_kernel_1d(
    kernel_size: int,
    norm: bool = False,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate Yang Hui triangle (Pascal's triangle) by a given number.

    Inspired by Kornia implementation.

    Parameters
    ----------
    kernel_size: height and width of the kernel.
    norm: if to normalize the kernel or not. Default: False.
    device: tensor device
    dtype: tensor dtype

    Returns
    -------
    kernel shaped as :math:`(kernel_size,)`

    Examples
    --------
    >>> get_pascal_kernel_1d(1)
    tensor([1.])
    >>> get_pascal_kernel_1d(2)
    tensor([1., 1.])
    >>> get_pascal_kernel_1d(3)
    tensor([1., 2., 1.])
    >>> get_pascal_kernel_1d(4)
    tensor([1., 3., 3., 1.])
    >>> get_pascal_kernel_1d(5)
    tensor([1., 4., 6., 4., 1.])
    >>> get_pascal_kernel_1d(6)
    tensor([ 1.,  5., 10., 10.,  5.,  1.])
    """
    pre: list[float] = []
    cur: list[float] = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = torch.tensor(cur, device=device, dtype=dtype)

    if norm:
        out = out / out.sum()

    return out


def get_pascal_kernel_nd(
    kernel_size: Union[Tuple[int, int], int],
    norm: bool = True,
    dim: int = 2,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate pascal filter kernel by kernel size.

    Inspired by Kornia implementation.

    Parameters
    ----------
    kernel_size: height and width of the kernel.
    norm: if to normalize the kernel or not. Default: True.
    device: tensor device
    dtype: tensor dtype

    Returns
    -------
    if kernel_size is an integer the kernel will be shaped as (kernel_size, kernel_size)
    otherwise the kernel will be shaped as kernel_size

    Examples
    --------
    >>> get_pascal_kernel_2d(1)
    tensor([[1.]])
    >>> get_pascal_kernel_2d(4)
    tensor([[0.0156, 0.0469, 0.0469, 0.0156],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0156, 0.0469, 0.0469, 0.0156]])
    >>> get_pascal_kernel_2d(4, norm=False)
    tensor([[1., 3., 3., 1.],
            [3., 9., 9., 3.],
            [3., 9., 9., 3.],
            [1., 3., 3., 1.]])
    """
    kernel_dims = _unpack_kernel_size(kernel_size, dim)

    kernel = [
        get_pascal_kernel_1d(kd, device=device, dtype=dtype) for kd in kernel_dims
    ]

    if dim == 2:
        kernel = kernel[0][:, None] * kernel[1][None, :]
    elif dim == 3:
        kernel = (
            kernel[0][:, None, None]
            * kernel[1][None, :, None]
            * kernel[2][None, None, :]
        )
    if norm:
        kernel = kernel / torch.sum(kernel)
    return kernel


def _max_blur_pool_by_kernel2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    stride: int,
    max_pool_size: int,
    ceil_mode: bool,
) -> torch.Tensor:
    """Compute max_blur_pool by a given :math:`CxC_(out, None)xNxN` kernel.

    Inspired by Kornia implementation.
    """
    # compute local maxima
    x = F.max_pool2d(
        x, kernel_size=max_pool_size, padding=0, stride=1, ceil_mode=ceil_mode
    )
    # blur and downsample
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]), dim=2)
    return F.conv2d(x, kernel, padding=padding, stride=stride, groups=x.size(1))


def _max_blur_pool_by_kernel3d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    stride: int,
    max_pool_size: int,
    ceil_mode: bool,
) -> torch.Tensor:
    """Compute max_blur_pool by a given :math:`CxC_(out, None)xNxNxN` kernel.

    Inspired by Kornia implementation.
    """
    # compute local maxima
    x = F.max_pool3d(
        x, kernel_size=max_pool_size, padding=0, stride=1, ceil_mode=ceil_mode
    )
    # blur and downsample
    padding = _compute_zero_padding(
        (kernel.shape[-3], kernel.shape[-2], kernel.shape[-1]), dim=3
    )
    return F.conv3d(x, kernel, padding=padding, stride=stride, groups=x.size(1))


class MaxBlurPool2D(nn.Module):
    """Compute pools and blurs and downsample a given feature map.

    Inspired by Kornia MaxBlurPool implementation. Equivalent to
    ```nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))```

    Parameters
    ----------
    kernel_size : the kernel size for max pooling. Tensor of shape (B, C, Y, X).
    stride: stride for pooling.
    max_pool_size: the kernel size for max pooling.
    ceil_mode: should be true to match output size of conv2d with same kernel size.

    Returns
    -------
    torch.Tensor: the transformed tensor of shape (B, C, Y / stride, X / stride)

    Examples
    --------
        >>> import torch.nn as nn
        >>> from kornia.filters.blur_pool import BlurPool2D
        >>> input = torch.eye(5)[None, None]
        >>> mbp = MaxBlurPool2D(kernel_size=3, stride=2,
        max_pool_size=2, ceil_mode=False)
        >>> mbp(input)
        tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
        >>> seq = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1),
        BlurPool2D(kernel_size=3, stride=2))
        >>> seq(input)
        tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
    """

    def __init__(
        self,
        kernel_size: Union[Tuple[int], int],
        stride: int = 2,
        max_pool_size: int = 2,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode
        self.kernel = get_pascal_kernel_nd(kernel_size, norm=True, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the function."""
        self.kernel = torch.as_tensor(self.kernel, device=x.device, dtype=x.dtype)
        return _max_blur_pool_by_kernel2d(
            x,
            self.kernel.repeat((x.size(1), 1, 1, 1)),
            self.stride,
            self.max_pool_size,
            self.ceil_mode,
        )


class MaxBlurPool3D(nn.Module):
    """Compute pools and blurs and downsample a given feature map.

    Inspired by Kornia MaxBlurPool implementation. Equivalent to
    ```nn.Sequential(nn.MaxPool3d(...), BlurPool3D(...))```

    Parameters
    ----------
    kernel_size : the kernel size for max pooling. Tensor of shape (B, C, Z, Y, X).
    stride: stride for pooling.
    max_pool_size: the kernel size for max pooling.
    ceil_mode: should be true to match output size of conv3d with same kernel size.

    Returns
    -------
    torch.Tensor: the transformed tensor of shape
    (B, C, Z / stride, Y / stride, X / stride)

    Examples
    --------
        >>> import torch.nn as nn
        >>> from kornia.filters.blur_pool import BlurPool3D
        >>> input = torch.eye(5)[None, None]
        >>> mbp = MaxBlurPool3D(kernel_size=3, stride=2,
        max_pool_size=2, ceil_mode=False)
        >>> mbp(input)
        tensor([[[[[0.5625, 0.3125],
                   [0.3125, 0.8750]]]]])
        >>> seq = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=1),
        BlurPool3D(kernel_size=3, stride=2))
        >>> seq(input)
        tensor([[[[[0.5625, 0.3125],
                   [0.3125, 0.8750]]]]])
    """

    def __init__(
        self,
        kernel_size: Union[Tuple[int], int],
        stride: int = 2,
        max_pool_size: int = 2,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode
        self.kernel = get_pascal_kernel_nd(kernel_size, norm=True, dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the function."""
        self.kernel = torch.as_tensor(self.kernel, device=x.device, dtype=x.dtype)
        return _max_blur_pool_by_kernel3d(
            x,
            self.kernel.repeat((x.size(1), 1, 1, 1, 1)),
            self.stride,
            self.max_pool_size,
            self.ceil_mode,
        )

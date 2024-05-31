"""Receptive field calculation for computing the tile overlap."""

# TODO better docstring and function names
# Adapted from: https://github.com/frgfm/torch-scan

# import math
# import warnings
# from typing import Tuple, Union

# from torch import Tensor, nn
# from torch.nn import Module
# from torch.nn.modules.batchnorm import _BatchNorm
# from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
# from torch.nn.modules.pooling import (
#     _AdaptiveAvgPoolNd,
#     _AdaptiveMaxPoolNd,
#     _AvgPoolNd,
#     _MaxPoolNd,
# )


# def module_rf(module: Module, inp: Tensor, out: Tensor) -> Tuple[float, float, float]:
#     """Estimate the spatial receptive field of the module.

#     Parameters
#     ----------
#     module : Module
#         Module to estimate the receptive field.
#     inp : Tensor
#         Input tensor.
#     out : Tensor
#         Output tensor.

#     Returns
#     -------
#     Tuple[float, float, float]
#         Receptive field, effective stride and padding.
#     """
#     if isinstance(
#         module,
#         (
#             nn.Identity,
#             nn.Flatten,
#             nn.ReLU,
#             nn.ELU,
#             nn.LeakyReLU,
#             nn.ReLU6,
#             nn.Tanh,
#             nn.Sigmoid,
#             _BatchNorm,
#             nn.Dropout,
#             nn.Linear,
#         ),
#     ):
#         return 1.0, 1.0, 0.0
#     elif isinstance(module, _ConvTransposeNd):
#         return rf_convtransposend(module, inp, out)
#     elif isinstance(module, (_ConvNd, _MaxPoolNd, _AvgPoolNd)):
#         return rf_aggregnd(module, inp, out)
#     elif isinstance(module, (_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd)):
#         return rf_adaptive_poolnd(module, inp, out)
#     else:
#         warnings.warn(
#             f"Module type not supported: {module.__class__.__name__}", stacklevel=1
#         )
#         return 1.0, 1.0, 0.0


# def rf_convtransposend(
#     module: _ConvTransposeNd, _: Tensor, __: Tensor
# ) -> Tuple[float, float, float]:
#     k = (
#         module.kernel_size[0]
#         if isinstance(module.kernel_size, tuple)
#         else module.kernel_size
#     )
#     s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
#     return -k, 1.0 / s, 0.0


# def rf_aggregnd(
#     module: Union[_ConvNd, _MaxPoolNd, _AvgPoolNd], _: Tensor, __: Tensor
# ) -> Tuple[float, float, float]:
#     k = (
#         module.kernel_size[0]
#         if isinstance(module.kernel_size, tuple)
#         else module.kernel_size
#     )
#     if hasattr(module, "dilation"):
#         d = (
#             module.dilation[0]
#             if isinstance(module.dilation, tuple)
#             else module.dilation
#         )
#         k = d * (k - 1) + 1
#     s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
#     p = module.padding[0] if isinstance(module.padding, tuple) else module.padding
#     return k, s, p  # type: ignore[return-value]


# def rf_adaptive_poolnd(
#     _: Union[_AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd], inp: Tensor, out: Tensor
# ) -> Tuple[int, int, float]:
#     stride = math.ceil(inp.shape[-1] / out.shape[-1])
#     kernel_size = stride
#     padding = (inp.shape[-1] - kernel_size * stride) / 2

#     return kernel_size, stride, padding

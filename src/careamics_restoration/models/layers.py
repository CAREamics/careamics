import torch
import torch.nn as nn

# TODO finish docstrings


class Conv_Block(nn.Module):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    conv_dim : _type_
        _description_
    in_channels : _type_
        _description_
    out_channels : _type_
        _description_
    intermediate_channel_multiplier : int, optional
        _description_, by default 1
    stride : int, optional
        _description_, by default 1
    padding : int, optional
        _description_, by default 1
    bias : bool, optional
        _description_, by default True
    groups : int, optional
        _description_, by default 1
    activation : str, optional
        _description_, by default "ReLU"
    dropout_perc : float, optional
        _description_, by default 0
    use_batch_norm : bool, optional
        _description_, by default False
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
        """Forward pass."""
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

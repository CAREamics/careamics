"""For creating networks that train independent models for each channel."""
from operator import getitem, setitem
from typing import Any, Dict, Type

import torch
import torch.nn as nn

from ..utils import get_logger

logger = get_logger(__name__)


class IndChannelModel(nn.Module):
    def __init__(
        self,
        model: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        in_channels_keyword: str = "in_channels",
        out_channels_keyword: str = "num_classes",
    ):
        """
        A class to create a model where the base model is duplicated for each
        input channel. Each model is trained seperately for each channel and
        the outputs are recomined at the end.

        Parameters
        ----------
        model : torch.nn.Module subclass
            The model class, which will be initialized with `model_kwargs`.
        model_kwargs : dict(str, any)
            Model keyword arguments
        in_channels_keyword : str, optional
            The argument that controls the input channels in the `model_kwargs`.
            default = 'in_channels'.

        Raises
        ------
        ValueError
            If the `in_channels_keyword` is not found in the
            `model_kwargs`.
        """
        super().__init__()

        # get the number of input channels
        try:
            in_channels = getitem(model_kwargs, in_channels_keyword)
        except AttributeError as e:
            raise ValueError(
                "Input channels keyword, '{}', not found in model_kwargs".format(
                    in_channels_keyword
                )
            ) from e

        # Create a copy of the model config but for a single channel
        model_config_single_channel = model_kwargs.copy()
        setitem(model_config_single_channel, in_channels_keyword, 1)
        setitem(model_config_single_channel, out_channels_keyword, 1)

        # Create a seperate model for each channel
        self.channel_models = nn.ModuleList(
            [model(**model_config_single_channel) for _ in range(in_channels)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Passes each channel to a seperate independent UNet model. Returns the
        3 outputs stacked together.

        Parameters
        ----------
        x :  torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the model.
        """
        # Alternative: pre-initialise 0-tensor and populate each channel
        y = []
        # Pass each channel to each model
        for i, model in enumerate(self.channel_models):
            y.append(model(x[:, [i], ...]))

        return torch.cat(y, dim=1)

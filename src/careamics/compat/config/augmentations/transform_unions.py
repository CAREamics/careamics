"""Type used to represent all transformations users can create."""

from typing import Annotated, Union

from pydantic import Discriminator

from careamics.config.augmentations import XYFlipConfig, XYRandomRotate90Config

from .normalize_config import NormalizeConfig

NORM_AND_SPATIAL_UNION = Annotated[
    Union[
        NormalizeConfig,
        XYFlipConfig,
        XYRandomRotate90Config,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""All transforms including normalization."""


SPATIAL_TRANSFORMS_UNION = Annotated[
    Union[
        XYFlipConfig,
        XYRandomRotate90Config,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""Available spatial transforms in CAREamics."""

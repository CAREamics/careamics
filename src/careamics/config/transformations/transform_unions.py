"""Type used to represent all transformations users can create."""

from typing import Annotated, Union

from pydantic import Discriminator

from .normalize_config import NormalizeConfig
from .xy_flip_config import XYFlipConfig
from .xy_random_rotate90_config import XYRandomRotate90Config

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

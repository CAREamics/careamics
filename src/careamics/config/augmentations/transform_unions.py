"""Type used to represent all transformations users can create."""

from typing import Annotated, Union

from pydantic import Discriminator

from .xy_flip_config import XYFlipConfig
from .xy_random_rotate90_config import XYRandomRotate90Config

SPATIAL_TRANSFORMS_UNION = Annotated[
    Union[
        XYFlipConfig,
        XYRandomRotate90Config,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""Available spatial transforms in CAREamics."""

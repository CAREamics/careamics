"""Type used to represent all transformations users can create."""

from typing import Annotated, Union

from pydantic import Discriminator

from .swap_input_target_config import SwapInputTargetConfig
from .tilt_axis_random_rotate90_config import TiltAxisRandomRotate90Config
from .xy_flip_config import XYFlipConfig
from .xy_random_rotate90_config import XYRandomRotate90Config

DefaultAugs = Annotated[
    Union[
        XYFlipConfig,
        XYRandomRotate90Config,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""Augmentations available for standard CAREamics spatial transforms."""

CryoCAREAugs = Annotated[
    Union[
        TiltAxisRandomRotate90Config,
        SwapInputTargetConfig,
    ],
    Discriminator("name"),
]
"""Augmentations available specifically for CryoCARE."""

AllAugs = Annotated[
    Union[
        XYFlipConfig,
        XYRandomRotate90Config,
        TiltAxisRandomRotate90Config,
        SwapInputTargetConfig,
    ],
    Discriminator("name"),
]
"""All augmentations executable by the dataset pipeline."""

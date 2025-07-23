"""Type used to represent all transformations users can create."""

from typing import Annotated, Union

from pydantic import Discriminator

from .normalize_models import NoNormModel, StandardizeModel
from .xy_flip_model import XYFlipModel
from .xy_random_rotate90_model import XYRandomRotate90Model

NORM_AND_SPATIAL_UNION = Annotated[
    Union[
        StandardizeModel,
        NoNormModel,
        XYFlipModel,
        XYRandomRotate90Model,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""All transforms including normalization."""


NORMALIZATION_UNION = Annotated[
    Union[
        StandardizeModel,
        NoNormModel,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]

SPATIAL_TRANSFORMS_UNION = Annotated[
    Union[
        XYFlipModel,
        XYRandomRotate90Model,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""Available spatial transforms in CAREamics."""

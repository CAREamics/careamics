"""Type used to represent all transformations users can create."""

from typing import Annotated, Union

from pydantic import Discriminator

from .n2v_manipulate_model import N2VManipulateModel
from .xy_flip_model import XYFlipModel
from .xy_random_rotate90_model import XYRandomRotate90Model

TRANSFORMS_UNION = Annotated[
    Union[
        XYFlipModel,
        XYRandomRotate90Model,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""Available transforms in CAREamics."""

N2V_TRANSFORMS_UNION = Annotated[
    Union[
        XYFlipModel,
        XYRandomRotate90Model,
        N2VManipulateModel,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]
"""Available N2V-compatible transforms in CAREamics."""

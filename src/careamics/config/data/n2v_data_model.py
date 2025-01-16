"""Noise2Void specific data configuration model."""

from collections.abc import Sequence

from pydantic import Field

from careamics.config.data.data_model import GeneralDataConfig
from careamics.config.transformations import (
    N2V_TRANSFORMS_UNION,
    XYFlipModel,
    XYRandomRotate90Model,
)


class N2VDataConfig(GeneralDataConfig):
    """N2V specific data configuration model."""

    transforms: Sequence[N2V_TRANSFORMS_UNION] = Field(
        default=[XYFlipModel(), XYRandomRotate90Model()],
        validate_default=True,
    )
    """N2V compatible transforms. N2VManpulate should be the last transform."""

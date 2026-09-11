"""Configuration classes for segmentation losses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _SegmentationLossConfig(BaseModel):
    """Common configuration for segmentation losses."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    class_weights: list[float] | None = None
    """Optional weight for each segmentation class."""


class DiceLossConfig(_SegmentationLossConfig):
    """Configuration for Dice loss."""

    name: Literal["dice"] = "dice"


class CELossConfig(_SegmentationLossConfig):
    """Configuration for cross-entropy loss."""

    name: Literal["ce"] = "ce"


class DiceCELossConfig(_SegmentationLossConfig):
    """Configuration for combined Dice and cross-entropy loss."""

    name: Literal["dice_ce"] = "dice_ce"
    dice_weight: float = Field(default=1.0, gt=0.0)
    ce_weight: float = Field(default=1.0, gt=0.0)

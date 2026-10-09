"""Losses Pydantic configurations."""

__all__ = [
    "CELossConfig",
    "DiceCELossConfig",
    "DiceLossConfig",
    "KLLossConfig",
    "LVAELossConfig",
    "SegmentatioLossConfig",
    "SegmentationLossConfig",
]

from .lvae_loss_config import (
    KLLossConfig,
    LVAELossConfig,
)
from .seg_loss_config import (
    CELossConfig,
    DiceCELossConfig,
    DiceLossConfig,
)

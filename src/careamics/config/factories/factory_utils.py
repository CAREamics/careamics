"""Utility to handle augmentation configuration."""

from collections.abc import Sequence
from typing import Literal

from careamics.config.augmentations import (
    SPATIAL_TRANSFORMS_UNION,
    XYFlipConfig,
    XYRandomRotate90Config,
)

from .data_factory import list_spatial_augmentations


def assemble_augmentations(
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None,
    seed: int | None = None,
) -> list[SPATIAL_TRANSFORMS_UNION]:
    """Assemble a list of augmentations.

    Parameters
    ----------
    augmentations : Sequence of {"x_flip", "y_flip", "rotate_90"} or None
        List of augmentations to apply. If `None`, all augmentations are applied.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    list of augmentations
        List of augmentations to apply.
    """
    augs: list[XYFlipConfig | XYRandomRotate90Config] | None = None
    if augmentations is not None:
        augs = []

        x_flip_present = "x_flip" in augmentations
        y_flip_present = "y_flip" in augmentations
        rotate_90_present = "rotate_90" in augmentations

        if x_flip_present or y_flip_present:
            augs.append(
                XYFlipConfig(
                    flip_x=x_flip_present,
                    flip_y=y_flip_present,
                    seed=seed,
                )
            )
        if rotate_90_present:
            augs.append(XYRandomRotate90Config(seed=seed))

    return list_spatial_augmentations(augs)

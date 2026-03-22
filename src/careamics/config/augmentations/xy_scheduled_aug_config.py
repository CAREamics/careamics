"""Pydantic model for the XYScheduledAugmentation transform."""

from typing import Literal

from pydantic import ConfigDict, Field

from .transform_config import TransformConfig


class XYScheduledAugConfig(TransformConfig):
    """
    Pydantic model for the deterministic dihedral-8 scheduled augmentation.

    When this config is included in ``NGDataConfig.augmentations``, the dataset
    will apply geometric transforms in a deterministic, epoch-aware cycle instead
    of random per-sample flips and rotations.

    The ``ScheduledAugCallback`` must be added to the Lightning ``Trainer``
    callbacks so that the transform's epoch counter is updated at the start of
    each training epoch.

    Attributes
    ----------
    name : Literal["XYScheduledAugmentation"]
        Name of the transformation (discriminator field).
    n_transforms : int
        Number of dihedral group elements to cycle through (1–8).  Default is 8,
        which uses the full dihedral group.  Smaller values use the first
        ``n_transforms`` entries in the group table:
        identity, rot90, rot180, rot270, flip_x, flip_y, rot90+flip_x,
        rot90+flip_y.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYScheduledAugmentation"] = "XYScheduledAugmentation"
    n_transforms: int = Field(
        default=8,
        ge=1,
        le=8,
        description="Number of dihedral ops to cycle through (1–8).",
    )

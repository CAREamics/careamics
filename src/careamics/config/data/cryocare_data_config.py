"""CryoCARE-specific data configuration."""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from careamics.config.augmentations import CryoCAREAugs

from .data_config import DataConfig
from .patching_strategies import StratifiedPatchingConfig


class CryoCAREDataConfig(DataConfig[CryoCAREAugs]):
    """Data configuration for CryoCARE training.

    CryoCARE operates on single-channel 3D tomograms and uses stratified sampling.
    Its input-target swapping and tilt-axis rotations are intentionally kept out of
    the generic :class:`DataConfig`.
    """

    tilt_axis: Literal["Y", "X", "Z"]
    """Physical axis about which the specimen was rotated during acquisition."""

    validation_fraction: float = Field(default=0.1, gt=0, lt=1.0)
    """Fraction of the reconstructed stack that should be used validation. Train and
    validation are split along the `tilt_axis`."""

    patching: StratifiedPatchingConfig
    """Stratified 3D patching required by CryoCARE."""

    channels: None = Field(default=None)
    """CryoCARE does not support channel selection."""

    augmentations: Sequence[CryoCAREAugs] = Field(default_factory=tuple)
    """CryoCARE-specific augmentations."""

    @field_validator("axes", mode="after")
    @classmethod
    def contains_z(cls, value: str) -> str:
        """Validate axes with "Z".

        Parameters
        ----------
        value : str
            Axes.

        Returns
        -------
        str
            Axes if they contain "Z".

        Raises
        ------
        ValueError
            If `axes` does not contain "Z".
        """
        if "Z" not in value:
            raise ValueError(f"Axes must contain `Z`, got {value}.")
        return value

    @field_validator("axes", mode="after")
    @classmethod
    def does_not_contain_c(cls, value: str) -> str:
        """Validate axes without "C".

        Parameters
        ----------
        value : str
            Axes.

        Returns
        -------
        str
            Axes if they do not contain "C".

        Raises
        ------
        ValueError
            If `axes` contains "C".
        """
        if "C" in value:
            raise ValueError(f"Axes must not contain `C`, got {value}.")
        return value

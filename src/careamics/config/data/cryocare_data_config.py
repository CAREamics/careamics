"""CryoCARE-specific data configuration."""

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from careamics.config.augmentations import (
    CryoCAREAugs,
    SwapInputTargetConfig,
    TiltAxisRandomRotate90Config,
)

from .data_config import DataConfig
from .patching_strategies import StratifiedPatchingConfig


def _rotation_is_compatible(
    tilt_axis: Literal["X", "Y", "Z"],
    patch_size: Sequence[int],
) -> bool:
    """Check whether a tilt-axis rotation can be applied to a patch.

    Parameters
    ----------
    tilt_axis : Literal["X", "Y", "Z"]
        Axis perpendicular to the rotation plane.
    patch_size : Sequence[int]
        Patch size.

    Returns
    -------
    bool
        Whether the two dimensions perpendicular to the tilt axis are equal.
    """
    spatial_patch_size = dict(zip("ZYX", patch_size, strict=True))
    rotation_axes = {
        "Z": ("Y", "X"),
        "Y": ("Z", "X"),
        "X": ("Z", "Y"),
    }[tilt_axis]
    return spatial_patch_size[rotation_axes[0]] == spatial_patch_size[rotation_axes[1]]


def _create_default_augmentations(
    tilt_axis: Literal["X", "Y", "Z"],
    patch_size: Sequence[int],
) -> list[CryoCAREAugs]:
    """Create default CryoCARE augmentations.

    Parameters
    ----------
    tilt_axis : Literal["X", "Y", "Z"]
        Tilt axis.
    patch_size : Sequence[int]
        Patch size.

    Returns
    -------
    list of CryoCARE augmentations
        Input-target swapping and, when compatible, tilt-axis rotation.
    """
    augmentations: list[CryoCAREAugs] = [SwapInputTargetConfig()]
    if _rotation_is_compatible(tilt_axis, patch_size):
        augmentations.insert(
            0,
            TiltAxisRandomRotate90Config(tilt_axis=tilt_axis),
        )

    return augmentations


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

    @model_validator(mode="before")
    @classmethod
    def set_default_augmentations(cls, value: Any) -> Any:
        """Set default augmentations when none were supplied.

        Parameters
        ----------
        value : Any
            Raw CryoCARE data configuration.

        Returns
        -------
        Any
            Raw configuration with default augmentations when applicable.
        """
        if not isinstance(value, dict) or "augmentations" in value:
            return value

        required = ("tilt_axis", "axes", "patching")
        if not all(key in value for key in required):
            return value

        patching = value["patching"]
        patch_size = (
            patching["patch_size"]
            if isinstance(patching, dict)
            else patching.patch_size
        )

        updated = value.copy()
        updated["augmentations"] = _create_default_augmentations(
            tilt_axis=value["tilt_axis"],
            patch_size=patch_size,
        )
        return updated

    @model_validator(mode="after")
    def validate_rotation_augmentations(self: Self) -> Self:
        """Validate configured tilt-axis rotations against the patch size.

        Returns
        -------
        Self
            Validated CryoCARE data configuration.

        Raises
        ------
        ValueError
            If a configured tilt-axis rotation is incompatible with the patch size.
        """
        for augmentation in self.augmentations:
            if isinstance(augmentation, TiltAxisRandomRotate90Config):
                if augmentation.tilt_axis != self.tilt_axis:
                    raise ValueError(
                        f"Tilt-axis is different in configuration (got "
                        f"{self.tilt_axis}) and in augmentation (got "
                        f"{augmentation.tilt_axis}."
                    )

                if not _rotation_is_compatible(
                    augmentation.tilt_axis,
                    self.patching.patch_size,
                ):
                    raise ValueError(
                        "TiltAxisRandomRotate90 requires equal patch dimensions "
                        f"perpendicular to tilt axis '{augmentation.tilt_axis}'."
                    )
        return self

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

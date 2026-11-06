"""CryoCARE-specific data configuration."""

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from careamics.config.augmentations import (
    CryoCAREAugs,
    SwapInputTargetConfig,
    TiltAxisRandomRotate90Config,
)

from .data_config import DataConfig, Mode
from .patching_strategies import StratifiedPatchingConfig


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
    spatial_patch_size = dict(zip("ZYX", patch_size, strict=True))
    rotation_axes = {
        "Z": ("Y", "X"),
        "Y": ("Z", "X"),
        "X": ("Z", "Y"),
    }[tilt_axis]

    augmentations: list[CryoCAREAugs] = [SwapInputTargetConfig()]
    if spatial_patch_size[rotation_axes[0]] == spatial_patch_size[rotation_axes[1]]:
        augmentations.insert(
            0,
            TiltAxisRandomRotate90Config(tilt_axis=tilt_axis),
        )

    return augmentations


class CryoCAREDataConfig(DataConfig[CryoCAREAugs]):
    """Data configuration for CryoCARE."""

    tilt_axis: Literal["Y", "X", "Z"]
    """Physical axis about which the specimen was rotated during acquisition."""

    validation_fraction: float = Field(default=0.1, gt=0, lt=1.0)
    """Fraction of the reconstructed stack that should be used validation. Train and
    validation are split along the `tilt_axis`."""

    augmentations: Sequence[CryoCAREAugs] = Field(default_factory=tuple)
    """CryoCARE-specific augmentations. This parameter is automatically populated and
    cannot be set by users. Any supplied value will overwritten."""

    @model_validator(mode="before")
    @classmethod
    def set_augmentations(cls, value: Any) -> Any:
        """Generate augmentations from the tilt axis and patch size.

        Parameters
        ----------
        value : Any
            Raw CryoCARE data configuration.

        Returns
        -------
        Any
            Raw configuration with generated augmentations when applicable.
        """
        if not isinstance(value, dict):
            return value

        if value.get("mode") != Mode.TRAINING:
            updated = value.copy()
            updated["augmentations"] = []
            return updated

        required = ("tilt_axis", "axes", "patching")
        if not all(key in value for key in required):
            return value

        patching = value["patching"]
        patch_size = (
            patching["patch_size"]
            if isinstance(patching, dict)
            else patching.patch_size
        )

        # overwrite augmentations to ensure compatibility between patch size, tilt axis,
        # and random rotation augmentations
        updated = value.copy()
        updated["augmentations"] = _create_default_augmentations(
            tilt_axis=value["tilt_axis"],
            patch_size=patch_size,
        )
        return updated

    @model_validator(mode="after")
    def validate_training_patching(self) -> "CryoCAREDataConfig":
        """Require stratified patching for CryoCARE training and validation.

        Returns
        -------
        CryoCAREDataConfig
            Validated CryoCARE data configuration.

        Raises
        ------
        ValueError
            If a training or validation configuration does not use stratified patching.
        """
        if self.mode in (Mode.TRAINING, Mode.VALIDATING) and not isinstance(
            self.patching, StratifiedPatchingConfig
        ):
            raise ValueError(
                "CryoCARE training and validation require stratified patching, got "
                f"'{self.patching.name}'."
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

    @field_validator("channels", mode="after")
    @classmethod
    def validate_channels(
        cls,
        value: Sequence[int] | None,
        info: ValidationInfo,
    ) -> Sequence[int] | None:
        """Validate that channel selection is disabled.

        Parameters
        ----------
        value : sequence of int or None
            Channel selection.
        info : ValidationInfo
            Validation information.

        Returns
        -------
        sequence of int or None
            The validated channel selection.

        Raises
        ------
        ValueError
            If channels are selected.
        """
        if value is not None:
            raise ValueError(
                "CryoCARE does not support channel selection, remove `channel` "
                "parameter."
            )
        return value

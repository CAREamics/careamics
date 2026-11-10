"""CryoCARE-specific data configuration."""

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from careamics.config.augmentations import (
    CryoCAREAugs,
    SwapInputTargetConfig,
    TiltAxisRandomRotate90Config,
)

from .data_config import (
    DataConfig,
    DataInputFormat,
    Mode,
)
from .patching_strategies import StratifiedPatchingConfig


def _create_augmentations(
    tilt_axis: Literal["X", "Y", "Z"],
    patch_size: Sequence[int],
) -> list[CryoCAREAugs]:
    """Create CryoCARE augmentations.

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
    cannot be set by users. Any supplied value will be overwritten."""

    # TODO in cryocare example, num validation patches created is 120
    # - are they all used in CSBDeep?
    # - does it matter to have as many?
    n_val_patches: int = Field(default=50, ge=20)
    """The number of patches to set aside for validation during training."""

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

        # before validator, so we need to check everyting is there
        # otherwise return and let normal validation fail
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
        updated["augmentations"] = _create_augmentations(
            tilt_axis=value["tilt_axis"],
            patch_size=patch_size,
        )
        return updated

    @model_validator(mode="after")
    def validate_training_patching(self) -> "CryoCAREDataConfig":
        """Require stratified patching for CryoCARE training.

        Returns
        -------
        CryoCAREDataConfig
            Validated CryoCARE data configuration.

        Raises
        ------
        ValueError
            If a training configuration does not use stratified patching.
        """
        if self.mode == Mode.TRAINING and not isinstance(
            self.patching, StratifiedPatchingConfig
        ):
            raise ValueError(
                "CryoCARE training requires stratified patching, got "
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

    @field_validator("target_axes", mode="after")
    @classmethod
    def target_axes_match_axes(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        """Validate that target axes match input axes.

        Parameters
        ----------
        value : str or None
            Target axes. ``None`` means that input axes are used.
        info : ValidationInfo
            Validation information containing the input axes.

        Returns
        -------
        str or None
            The validated target axes.

        Raises
        ------
        ValueError
            If target axes differ from input axes.
        """
        if value is not None and value != info.data["axes"]:
            raise ValueError(
                "CryoCARE requires target axes to match input axes, got "
                f"target axes '{value}' and input axes '{info.data['axes']}'."
            )
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

    def convert_mode(
        self: Self,
        new_mode: Literal["validating", "predicting"],
        new_patch_size: Sequence[int] | None = None,
        overlap_size: Sequence[int] | None = None,
        new_batch_size: int | None = None,
        new_data_type: DataInputFormat | None = None,
        new_axes: str | None = None,
        new_target_axes: str | None = None,
        new_channels: Sequence[int] | Literal["all"] | None = None,
        new_in_memory: bool | None = None,
        new_dataloader_params: dict[str, Any] | None = None,
    ) -> "CryoCAREDataConfig":
        """Convert mode while preserving CryoCARE-specific fields.

        Parameters
        ----------
        new_mode : Literal["validating", "predicting"]
            The new dataset mode.
        new_patch_size : Sequence[int] or None, default=None
            New patch size for validation or tiled prediction.
        overlap_size : Sequence[int] or None, default=None
            Tile overlap, required when ``new_patch_size`` is specified for prediction.
        new_batch_size : int or None, default=None
            New batch size.
        new_data_type : DataInputFormat or None, default=None
            New data type.
        new_axes : str or None, default=None
            New axes are not permitted.
        new_target_axes : str or None, default=None
            New target axes are not permitted.
        new_channels : Sequence[int], "all" or None, default=None
            New channel selection is not permitted.
        new_in_memory : bool or None, default=None
            New in-memory loading setting.
        new_dataloader_params : dict of {str: Any} or None, default=None
            New dataloader parameters.

        Returns
        -------
        CryoCAREDataConfig
            Converted configuration with CryoCARE-specific fields preserved.
        """
        unsupported_changes = {
            "new_axes": new_axes,
            "new_target_axes": new_target_axes,
            "new_channels": new_channels,
        }
        for name, value in unsupported_changes.items():
            if value is not None:
                raise ValueError(
                    f"CryoCARE does not support changing `{name}` during conversion."
                )

        model_dict = self._get_converted_model_dict(
            new_mode=new_mode,
            new_patch_size=new_patch_size,
            overlap_size=overlap_size,
            new_batch_size=new_batch_size,
            new_data_type=new_data_type,
            new_axes=new_axes,
            new_target_axes=new_target_axes,
            new_channels=new_channels,
            new_in_memory=new_in_memory,
            new_dataloader_params=new_dataloader_params,
        )

        # add back the cryoCARE specific parameters
        model_dict.update(
            {
                "tilt_axis": self.tilt_axis,
                "validation_fraction": self.validation_fraction,
            }
        )
        return CryoCAREDataConfig(**model_dict)

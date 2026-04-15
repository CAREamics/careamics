"""Data configuration."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Sequence
from enum import StrEnum
from pprint import pformat
from typing import Annotated, Any, Literal, Self, Union
from warnings import warn

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    field_validator,
    model_validator,
)

from careamics.config.augmentations import XYFlipConfig, XYRandomRotate90Config
from careamics.config.support import SupportedData
from careamics.config.utils.random import generate_random_seed
from careamics.config.validators import check_axes_validity, check_czi_axes_validity

from .normalization_config import NormalizationConfig
from .patch_filter import (
    MaskFilterConfig,
    MaxFilterConfig,
    MeanSTDFilterConfig,
    ShannonFilterConfig,
)
from .patching_strategies import (
    FixedRandomPatchingConfig,
    RandomPatchingConfig,
    StratifiedPatchingConfig,
    TiledPatchingConfig,
    WholePatchingConfig,
)

# TODO: Validate the specific sizes of tiles and overlaps given UNet constraints
#   - needs to be done in the Configuration
#   - patches and overlaps sizes must also be checked against dimensionality
#   - Should we have a UNet and a LVAE DataConfig subclass with specific validations?

# TODO: is 3D updated anywhere in the code in CAREamist/downstream?
#       - this will be important when swapping the data config in Configuration
#       - `set_3D` currently not implemented here

# TODO: this module is very long, can we split the validation somewhere else and
#       leverage Pydantic to add validation directly to the declaration of each field?


def _is_3D(axes: str, data_type: SupportedData) -> bool:
    """Determine whether the `axes` and `data_type` combination specifies 3D data.

    Parameters
    ----------
    axes : str
        The axes of the data.
    data_type : SupportedData
        The data format.

    Returns
    -------
    bool
        Whether the parameters specify 3D data.
    """
    if data_type != SupportedData.CZI:
        return "Z" in axes
    else:
        return ("Z" in axes) or ("T" in axes)


def _are_spatial_dims_maintained(
    old_data_type: SupportedData,
    old_axes: str,
    new_data_type: SupportedData,
    new_axes: str,
) -> bool:
    """Check that spatial dimensions are maintained between sets of data type and axes.

    Note that in the case of CZI, 3D can be both due to `T` and `Z` axes. We consider
    that spatial dims are maintained if converting between CZI `SCTYX` axes and `ZYX` in
    non-CZI format (in effect hoping that `T` has been relabelled as `Z` in the non-CZI
    format).

    Parameters
    ----------
    old_data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        Original data type.
    old_axes : str
        Original axes.
    new_data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        New data type.
    new_axes : str
        New axes.

    Returns
    -------
    bool
        Whether spatial dimensions are maintained.
    """
    is_3D_old = _is_3D(old_axes, old_data_type)
    is_3D_new = _is_3D(new_axes, new_data_type)

    if old_data_type == new_data_type and new_data_type == "czi":
        # for CZI data, check that Z did not switch to T or inversely
        if is_3D_old and is_3D_new:
            return ("Z" in old_axes and "Z" in new_axes) or (
                "T" in old_axes and "T" in new_axes
            )

    return is_3D_old == is_3D_new


def _validate_channel_conversion(
    old_axes: str,
    old_channels: Sequence[int] | None,
    new_axes: str,
    new_channels: Sequence[int] | None,
) -> None:
    """Validate the channel conversion.

    Parameters
    ----------
    old_axes : str
        Original axes.
    old_channels : Sequence[int] or None
        Original channels.
    new_axes : str
        New axes.
    new_channels : Sequence[int] or None
        New channels.

    Raises
    ------
    ValueError
        If the channel conversion is not valid.
    """
    # if switching C axis:
    # - removing C: original channels can be `None`, singleton or multiple. New
    #   channels can be `None` if original were `None` or singleton, but not
    #   multiple.
    # - adding C: original channels can only be `None`. New channels can be `None`
    #   (but we warn users that they need to have a singleton C axis in the data),
    #   or singleton, but not multiple.
    adding_C_axis = ("C" in new_axes) and ("C" not in old_axes)
    removing_C_axis = ("C" not in new_axes) and ("C" in old_axes)
    prev_channels_not_singleton = old_channels is not None and (len(old_channels) != 1)

    if adding_C_axis:
        if new_channels is None:
            warn(
                f"When switching to axes with 'C' (got {new_axes}) from axes "
                f"{old_axes}, errors may be raised in the model if the channel "
                f"dimension in the data is not a singleton dimension. To select a "
                f"specific channel, use the `new_channels` parameter (e.g. "
                f"`new_channels=[1]`).",
                stacklevel=1,
            )
        elif len(new_channels) != 1:
            raise ValueError(
                f"Cannot switch to axes with 'C' (got {new_axes}) from axes "
                f"{old_axes}, select a single channel using the `new_channels` "
                f"parameter (got channels {new_channels})."
            )
    elif removing_C_axis and prev_channels_not_singleton:
        raise ValueError(
            f"Cannot switch to axes without 'C' (got {new_axes}) from axes "
            f"{old_axes} when multiple channels were originally specified "
            f"({old_channels})."
        )

    # different number of channels
    if old_channels is not None and new_channels is not None:
        if len(new_channels) != len(old_channels):
            raise ValueError(
                f"Cannot switch between axes with different number of channels. "
                f"New channels length ({len(new_channels)}) does not match "
                f"current channels length ({len(old_channels)})."
            )
    elif old_channels is None and new_channels is not None:
        warn(
            f"Switching from all channels (`channels=None`) to specifying channels "
            f"{new_channels} may lead to errors if {new_channels} are not covering "
            f"all channels.",
            stacklevel=1,
        )  # Note that in the opposite case, old_channels is kept because
        # new_channels is None


def np_float_to_scientific_str(x: float) -> str:
    """Return a string scientific representation of a float.

    In particular, this method is used to serialize floats to strings, allowing
    numpy.float32 to be passed in the Pydantic model and written to a yaml file as str.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    str
        Scientific string representation of the input value.
    """
    return np.format_float_scientific(x, precision=7)


def get_default_num_workers() -> int:
    """Return the default number of dataloader workers for the current platform.

    Defaults by platform (benchmarked on BSD68, may need revisiting for larger datasets
    or more performant machines):
    - pytest: 0 - avoids multiprocessing overhead in tests.
    - Windows: 0 - multiprocessing with spawn is unreliable in dataloaders.
    - macOS: 0 - spawn-based worker init causes ~1 min startup hang even for a
      small number of workers, the throughput gain does not justify the wait.
    - Linux: min(cpu_count - 1, 4) - one core is left free to keep the UI
      responsive when training inside napari. Performance gains plateau around 4
      workers, so we cap there to avoid wasting resources.


    Returns
    -------
    int
        Default number of dataloader workers.
    """
    if "pytest" in sys.modules:
        return 0

    if platform.system() in ("Windows", "Darwin"):
        return 0

    max_workers = 4
    available_workers = os.cpu_count() or 0
    available_workers = max(0, available_workers - 1)
    return min(available_workers, max_workers)


Float = Annotated[float, PlainSerializer(np_float_to_scientific_str, return_type=str)]
"""Annotated float type, used to serialize floats to strings."""

PatchingConfig = Union[
    FixedRandomPatchingConfig,
    RandomPatchingConfig,
    StratifiedPatchingConfig,
    TiledPatchingConfig,
    WholePatchingConfig,
]
"""Patching strategy type."""

PatchFilterConfig = Union[
    MaxFilterConfig,
    MeanSTDFilterConfig,
    ShannonFilterConfig,
]
"""Patch filter type."""


class Mode(StrEnum):
    """Dataset mode."""

    TRAINING = "training"
    VALIDATING = "validating"
    PREDICTING = "predicting"


def default_in_memory(validated_params: dict[str, Any]) -> bool:
    """Default factory for the `in_memory` field.

    Based on the value of `data_type`, set the default for `in_memory` to `True` if
    the data type is 'array', 'tiff', or 'custom', and to `False` otherwise (`zarr`
    or 'czi').

    Parameters
    ----------
    validated_params : dict of {str: Any}
        Validated parameters.

    Returns
    -------
    bool
        Default value for the `in_memory` field.
    """
    return validated_params.get("data_type") not in ("zarr", "czi")


def _create_mask_filter(validated_params: dict[str, Any]) -> MaskFilterConfig | None:
    """Create a mask filter with auto-calculated coverage based on dimensionality.

    Parameters
    ----------
    validated_params : dict of {str: Any}
        Validated parameters containing 'mode', 'data_type', and 'axes'.

    Returns
    -------
    MaskFilterConfig | None
        Mask filter with auto-calculated coverage if in TRAINING mode, None otherwise.
    """
    mode = validated_params.get("mode")
    data_type = validated_params.get("data_type")
    axes = validated_params.get("axes", "")

    # only create mask filter in training mode
    if mode != Mode.TRAINING:
        return None

    # determine if data is 3D
    assert data_type is not None and isinstance(data_type, str)
    is_3d = _is_3D(axes, SupportedData(data_type))

    ndims = 3 if is_3d else 2
    coverage = 1 / (2**ndims)

    return MaskFilterConfig(coverage=coverage)


class DataConfig(BaseModel):
    """Next-Generation Dataset configuration.

    DataConfig are used for both training and prediction, with the patching strategy
    determining how the data is processed. Note that `random` is the only patching
    strategy compatible with training, while `tiled` and `whole` are only used for
    prediction.

    All supported transforms are defined in the SupportedTransform enum.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Dataset configuration
    mode: Mode
    """Dataset mode, either training, validating or predicting."""

    data_type: Literal["array", "tiff", "zarr", "czi", "custom"]
    """Type of input data."""

    axes: str
    """Axes of the data, as defined in SupportedAxes."""

    # TODO: update docs for stratified patching
    patching: PatchingConfig = Field(..., discriminator="name")
    """Patching strategy to use. Note that `random` is the only supported strategy for
    training, while `tiled` and `whole` are only used for prediction."""

    normalization: NormalizationConfig = Field(...)
    """Normalization configuration to use."""

    # Optional fields
    batch_size: int = Field(default=1, ge=1, validate_default=True)
    """Batch size for training."""

    in_memory: bool = Field(default_factory=default_in_memory, validate_default=True)
    """Whether to load all data into memory. This is only supported for 'array',
    'tiff' and 'custom' data types. Must be `True` for `array`. If `None`, defaults to
    `True` for 'array', 'tiff' and `custom`, and `False` for 'zarr' and 'czi' data
    types."""

    n_val_patches: int = Field(default=8, ge=0, validate_default=True)
    """The number of patches to set aside for validation during training. This parameter
    will be ignored if separate validation data is specified for training."""

    channels: Sequence[int] | None = Field(default=None)
    """Channels to use from the data. If `None`, all channels are used. Note that it is
    applied to both inputs and targets."""

    patch_filter: PatchFilterConfig | None = Field(default=None, discriminator="name")
    """Patch filter to apply when using random patching. Only available if
    mode is `training`."""

    mask_filter: MaskFilterConfig | None = Field(
        default_factory=lambda data: _create_mask_filter(data)
    )
    """Mask filter configuration to apply when using a mask during training.
    Coverage is automatically set to 1/(2**ndims) based on data dimensionality
    where ndims is determined from axes. Only available in `training` mode."""

    augmentations: Sequence[Union[XYFlipConfig, XYRandomRotate90Config]] = Field(
        default=(
            XYFlipConfig(),
            XYRandomRotate90Config(),
        ),
        validate_default=True,
    )
    """List of augmentations to apply to the data, available transforms are defined
    in SupportedTransform."""

    train_dataloader_params: dict[str, Any] = Field(
        default={"shuffle": True}, validate_default=True
    )
    """Dictionary of PyTorch training dataloader parameters. The dataloader parameters,
    should include the `shuffle` key, which is set to `True` by default. We strongly
    recommend to keep it as `True` to ensure the best training results."""

    val_dataloader_params: dict[str, Any] = Field(default={})
    """Dictionary of PyTorch validation dataloader parameters."""

    pred_dataloader_params: dict[str, Any] = Field(default={})
    """Dictionary of PyTorch prediction dataloader parameters."""

    num_workers: int = Field(default_factory=get_default_num_workers, ge=0)
    """Default number of workers for all dataloaders that do not explicitly set
    `num_workers`. Automatically detected based on the current platform:
    0 on Windows and macOS, `min(cpu_count - 1, 4)` on Linux."""

    seed: int = Field(default_factory=generate_random_seed, gt=0)
    """Random seed for reproducibility. If not specified, a random seed is generated."""

    @field_validator("axes")
    @classmethod
    def axes_valid(cls, axes: str, info: ValidationInfo) -> str:
        """
        Validate axes.

        Axes must:
        - be a combination of 'STCZYX'
        - not contain duplicates
        - contain at least 2 contiguous axes: X and Y
        - contain at most 4 axes
        - not contain both S and T axes

        Parameters
        ----------
        axes : str
            Axes to validate.
        info : ValidationInfo
            Validation information.

        Returns
        -------
        str
            Validated axes.

        Raises
        ------
        ValueError
            If axes are not valid.
        """
        if "data_type" not in info.data:
            raise ValueError(
                "Validation for `data_type` may have failed. Check for typos or "
                "missing field."
            )

        # Additional validation for CZI files
        if info.data["data_type"] == "czi":
            if not check_czi_axes_validity(axes):
                raise ValueError(
                    f"Invalid axes '{axes}'. Axes must be in the "
                    f"`SC(Z/T)YX` format, where Z or T are optional, and S and C can be"
                    f" singleton dimensions, but must be provided."
                )
        else:
            check_axes_validity(axes)

        return axes

    @field_validator("in_memory")
    @classmethod
    def validate_in_memory_with_data_type(cls, in_memory: bool, info: Any) -> bool:
        """
        Validate that in_memory is compatible with data_type.

        `in_memory` can only be True for 'array', 'tiff' and 'custom' data types.

        Parameters
        ----------
        in_memory : bool
            Whether to load data into memory.
        info : Any
            Additional information about the field being validated.

        Returns
        -------
        bool
            Validated in_memory value.

        Raises
        ------
        ValueError
            If in_memory is True for unsupported data types.
        """
        data_type = info.data.get("data_type")

        if in_memory and data_type in ("czi", "zarr"):
            raise ValueError(f"`in_memory` not supported for `data_type` {data_type}.")

        if not in_memory and data_type == "array":
            raise ValueError('`in_memory` must be True for "array" `data_type`.')

        return in_memory

    @field_validator("channels", mode="before")
    @classmethod
    def validate_channels(
        cls,
        channels: Sequence[int] | None,
        info: ValidationInfo,
    ) -> Sequence[int] | None:
        """
        Validate channels.

        Channels must be a sequence of non-negative integers without duplicates. If
        channels are not `None`, then `C` must be present in the axes.

        Parameters
        ----------
        channels : Sequence of int or None
            Channels to validate.
        info : ValidationInfo
            Validation information.

        Returns
        -------
        Sequence of int or None
            Validated channels.

        Raises
        ------
        ValueError
            If channels are not valid.
        """
        if channels is not None:
            if "C" not in info.data["axes"]:
                raise ValueError(
                    "Channels must be `None` if 'C' is not present in `axes`."
                )

            if isinstance(channels, int):
                channels = [channels]

            if not isinstance(channels, Sequence):
                raise ValueError("Channels must be a sequence of integers.")

            if len(channels) == 0:
                return None

            if not all(isinstance(ch, int) for ch in channels):
                raise ValueError("Channels must be integers.")

            if any(ch < 0 for ch in channels):
                raise ValueError("Channels must be non-negative integers.")

            if len(set(channels)) != len(channels):
                raise ValueError("Channels must not contain duplicates.")
        return channels

    @field_validator("patching")
    @classmethod
    def validate_patching_strategy_against_mode(
        cls, patching: PatchingConfig, info: ValidationInfo
    ) -> PatchingConfig:
        """
        Validate that the patching strategy is compatible with the dataset mode.

        - If mode is `training`, patching strategy must be `random` or `stratified`.
        - If mode is `validating`, patching must be `fixed_random`.
        - If mode is `predicting`, patching strategy must be `tiled` or `whole`.

        Parameters
        ----------
        patching : PatchingStrategies
            Patching strategy to validate.
        info : ValidationInfo
            Validation information.

        Returns
        -------
        PatchingStrategies
            Validated patching strategy.

        Raises
        ------
        ValueError
            If the patching strategy is not compatible with the dataset mode.
        """
        mode = info.data["mode"]
        if mode == Mode.TRAINING:
            if patching.name not in ["random", "stratified"]:
                raise ValueError(
                    f"Patching strategy '{patching.name}' is not compatible with "
                    f"mode '{mode.value}'. Use 'stratified' or 'random' for training."
                )
        elif mode == Mode.VALIDATING:
            if patching.name != "fixed_random":
                raise ValueError(
                    f"Patching strategy '{patching.name}' is not compatible with "
                    f"mode '{mode.value}'. Use 'fixed_random' for validating."
                )
        elif mode == Mode.PREDICTING:
            if patching.name not in ["tiled", "whole"]:
                raise ValueError(
                    f"Patching strategy '{patching.name}' is not compatible with "
                    f"mode '{mode.value}'. Use 'tiled' or 'whole' for predicting."
                )
        return patching

    @field_validator("patch_filter", "mask_filter")
    @classmethod
    def validate_filters_against_mode(
        cls,
        filter_obj: PatchFilterConfig | MaskFilterConfig | None,
        info: ValidationInfo,
    ) -> PatchFilterConfig | MaskFilterConfig | None:
        """
        Validate that the filters are only used during training.

        Parameters
        ----------
        filter_obj : PatchFilterConfig | MaskFilterConfig | None
            Filter to validate.
        info : ValidationInfo
            Validation information.

        Returns
        -------
        PatchFilterConfig | MaskFilterConfig | None
            Validated filter.

        Raises
        ------
        ValueError
            If a filter is used in a mode other than training.
        """
        mode = info.data["mode"]
        if filter_obj is not None and mode != Mode.TRAINING:
            raise ValueError(
                f"Filtering '{filter_obj.name}' only allowed in 'training' mode, "
                f"got mode '{mode.value}'."
            )
        return filter_obj

    @field_validator(
        "train_dataloader_params",
        "val_dataloader_params",
        "pred_dataloader_params",
        mode="after",
    )
    @classmethod
    def batch_size_not_in_dataloader_params(
        cls, dataloader_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate that `batch_size` is not set in the dataloader parameters.

        `batch_size` must be set through `batch_size` field, not
        through the dataloader parameters.

        Parameters
        ----------
        dataloader_params : dict of {str: Any}
            The dataloader parameters.

        Returns
        -------
        dict of {str: Any}
            The validated dataloader parameters.

        Raises
        ------
        ValueError
            If `batch_size` is present in the dataloader parameters.
        """
        if "batch_size" in dataloader_params:
            raise ValueError(
                "`batch_size` should not be set in the dataloader parameters. "
                "Use the `batch_size` field of `DataConfig` instead."
            )
        return dataloader_params

    @field_validator("train_dataloader_params")
    @classmethod
    def shuffle_train_dataloader(
        cls, train_dataloader_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate that "shuffle" is included in the training dataloader params.

        A warning will be raised if `shuffle=False`.

        Parameters
        ----------
        train_dataloader_params : dict of {str: Any}
            The training dataloader parameters.

        Returns
        -------
        dict of {str: Any}
            The validated training dataloader parameters.

        Raises
        ------
        ValueError
            If "shuffle" is not included in the training dataloader params.
        """
        if "shuffle" not in train_dataloader_params:
            raise ValueError(
                "`train_dataloader_params` must include the `shuffle` parameter."
            )
        elif ("shuffle" in train_dataloader_params) and (
            not train_dataloader_params["shuffle"]
        ):
            warn(
                "`train_dataloader_params` includes `shuffle=False`, which may lead to "
                "lower quality results.",
                stacklevel=1,
            )
        return train_dataloader_params

    @model_validator(mode="after")
    def validate_dimensions(self: Self) -> Self:
        """
        Validate 2D/3D dimensions between axes and patch size.

        Returns
        -------
        Self
            Validated data model.

        Raises
        ------
        ValueError
            If the patch size dimension is not compatible with the axes.
        """
        # "whole" patching does not have dimensions to validate
        if not hasattr(self.patching, "patch_size"):
            return self

        if self.data_type == "czi":
            # Z and T are both depth axes for CZI data
            expected_dims = 3 if ("Z" in self.axes or "T" in self.axes) else 2
            additional_message = " (`Z` and `T` are depth axes for CZI data)"
        else:
            expected_dims = 3 if "Z" in self.axes else 2
            additional_message = ""

        # infer dimension from requested patch size
        actual_dims = len(self.patching.patch_size)
        if actual_dims != expected_dims:
            raise ValueError(
                f"`patch_size` in `patching` must have {expected_dims} dimensions, "
                f"got {self.patching.patch_size} with axes {self.axes}"
                f"{additional_message}."
            )

        return self

    @model_validator(mode="after")
    def propagate_seed_to_augmentations(self: Self) -> Self:
        """
        Propagate the main seed to all augmentations that support seeds.

        This ensures that all augmentations use the same seed for
         reproducibility, unless they already have a seed explicitly set.

        Returns
        -------
        Self
            Data model with propagated seeds.
        """
        if self.seed is not None:
            for transform in self.augmentations:
                if hasattr(transform, "seed") and transform.seed is None:
                    transform.seed = self.seed
        return self

    @model_validator(mode="after")
    def propagate_seed_to_patching(self: Self) -> Self:
        """
        Propagate the main seed to the patching strategy if it supports seeds.

        This ensures that the patching strategy uses the same seed for reproducibility,
        unless it already has a seed explicitly set.

        Returns
        -------
        Self
            Data model with propagated seed.
        """
        if self.seed is not None:
            if hasattr(self.patching, "seed") and self.patching.seed is None:
                self.patching.seed = self.seed
        return self

    @field_validator("train_dataloader_params", "val_dataloader_params", mode="before")
    @classmethod
    def set_default_pin_memory(
        cls, dataloader_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Set default pin_memory for dataloader parameters if not provided.

        - If 'pin_memory' is not set, it defaults to True if CUDA is available.

        Parameters
        ----------
        dataloader_params : dict of {str: Any}
            The dataloader parameters.

        Returns
        -------
        dict of {str: Any}
            The dataloader parameters with pin_memory default applied.
        """
        if "pin_memory" not in dataloader_params:
            import torch

            dataloader_params["pin_memory"] = torch.cuda.is_available()
        return dataloader_params

    @model_validator(mode="after")
    def warn_inconsistent_num_workers(self: Self) -> Self:
        """Warn if `num_workers` conflicts with a per-dataloader value.

        This validator runs before ``set_default_workers_in_dataloaders``, so
        the dataloader dicts only contain user-supplied values at this point.
        Only fires when `num_workers` was explicitly set on the model.

        Returns
        -------
        Self
            Unchanged data model.
        """
        if "num_workers" not in self.model_fields_set:
            return self
        for name, params in (
            ("train_dataloader_params", self.train_dataloader_params),
            ("val_dataloader_params", self.val_dataloader_params),
            ("pred_dataloader_params", self.pred_dataloader_params),
        ):
            if "num_workers" in params and params["num_workers"] != self.num_workers:
                print(
                    f"Warning: `num_workers={self.num_workers}` conflicts with "
                    f"`{name}['num_workers']={params['num_workers']}`. "
                    f"The per-dataloader value takes precedence."
                )
        return self

    @model_validator(mode="after")
    def set_default_workers_in_dataloaders(self: Self) -> Self:
        """Set `num_workers` and `persistent_workers` defaults in all dataloaders.

        For each of `train_dataloader_params`, `val_dataloader_params`, and
        `pred_dataloader_params`: sets `num_workers` from the `num_workers`
        field if not already present, and sets ``persistent_workers=True`` when
        ``num_workers > 0`` and not already specified.

        Returns
        -------
        Self
            Validated data model with worker defaults applied to all dataloaders.
        """
        for params in (
            self.train_dataloader_params,
            self.val_dataloader_params,
            self.pred_dataloader_params,
        ):
            if "num_workers" not in params:
                params["num_workers"] = self.num_workers
            if "persistent_workers" not in params and params["num_workers"] > 0:
                params["persistent_workers"] = True
        return self

    def __str__(self) -> str:
        """
        Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def is_3D(self) -> bool:
        """
        Check if the data is 3D based on the axes.

        Either "Z" is in the axes and patching `patch_size` has 3 dimensions, or for CZI
        data, "Z" is in the axes or "T" is in the axes and patching `patch_size` has
        3 dimensions.

        This method is used during Configuration validation to cross checks dimensions
        with the algorithm configuration.

        Returns
        -------
        bool
            True if the data is 3D, False otherwise.
        """
        return _is_3D(self.axes, SupportedData(self.data_type))

    def set_3D(self, axes: str, patch_size: list[int]) -> None:
        """
        Set 3D parameters.

        Parameters
        ----------
        axes : str
            Axes.
        patch_size : list of int
            Patch size.
        """
        if not isinstance(self.patching, WholePatchingConfig):
            self.patching.patch_size = patch_size
        self.axes = axes

    # TODO: if switching from a state in which in_memory=True to an incompatible state
    # an error will be raised. Should that automatically be set to False instead?
    # TODO this method could be private and we could have public `to_validation_config`
    #   and `to_prediction_config` methods with appropriate parameters
    def convert_mode(
        self,
        new_mode: Literal["validating", "predicting"],
        new_patch_size: Sequence[int] | None = None,
        overlap_size: Sequence[int] | None = None,
        new_batch_size: int | None = None,
        new_data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        new_axes: str | None = None,
        new_channels: Sequence[int] | Literal["all"] | None = None,
        new_in_memory: bool | None = None,
        new_dataloader_params: dict[str, Any] | None = None,
    ) -> DataConfig:
        """
        Convert a training dataset configuration to a different mode.

        This method is intended to facilitate creating validation or prediction
        configurations from a training configuration.

        To perform tile prediction when switching to `predicting` mode, please provide
        both `new_patch_size` and `overlap_size`. Switching mode to `predicting` without
        specifying `new_patch_size` and `overlap_size` will apply the default patching
        strategy, namely `whole` image strategy. `new_patch_size` and `overlap_size` are
        only used when switching to `predicting`.

        `channels=None` will retain the same channels as in the current configuration.
        To select all channels, please specify all channels explicitly or pass
        `channels='all'`.

        New dataloader parameters will be placed in the appropriate dataloader params
        field depending on the new mode.

        To create a new training configuration, please use
        `careamics.config.create_ng_data_configuration`.

        This method compares the new parameters with the current ones and raises
        errors if incompatible changes are requested, such as switching between 2D and
        3D axes, or changing the number of channels. Incompatibility across parameters
        may be delegated to Pydantic validation.

        Parameters
        ----------
        new_mode : Literal["validating", "predicting"]
            The new dataset mode, one of `validating` or `predicting`.
        new_patch_size : Sequence of int, default=None
            New patch size. If None for `predicting`, uses default whole image strategy.
        overlap_size : Sequence of int, default=None
            New overlap size. Necessary when switching to `predicting` with tiled
            patching.
        new_batch_size : int, default=None
            New batch size.
        new_data_type : Literal['array', 'tiff', 'zarr', 'czi', 'custom'], default=None
            New data type.
        new_axes : str, default=None
            New axes.
        new_channels : Sequence of int or "all", default=None
            New channels.
        new_in_memory : bool, default=None
            New in_memory value.
        new_dataloader_params : dict of {str: Any}, default=None
            New dataloader parameters. These will be placed in the
            appropriate dataloader params field depending on the new mode.

        Returns
        -------
        DataConfig
            New DataConfig with the updated mode and parameters.

        Raises
        ------
        ValueError
            If conversion to training mode is requested, or if incompatible changes
            are requested.
        """
        if self.mode != Mode.TRAINING:
            raise ValueError(
                f"Conversion from mode '{self.mode}' to '{new_mode}' is not supported. "
                f"Only conversion from 'training' mode is supported."
            )
        if new_mode == Mode.TRAINING:
            raise ValueError(
                "Conversion to 'training' mode is not supported. Create a new "
                "DataConfig instead, for instance using "
                "`create_ng_data_configuration`."
            )

        # sanity checks
        # switching spatial axes
        if not _are_spatial_dims_maintained(
            SupportedData(self.data_type),
            self.axes,
            SupportedData(new_data_type or self.data_type),
            new_axes or self.axes,
        ):  # switching 2D/3D
            additional_msg = ""
            if self.data_type == "czi" or new_data_type == "czi":
                additional_msg = " Note that for CZI data, Z and T are both depth axes."

            raise ValueError(
                "Conversion between different spatial dimensions is not allowed. Got "
                f"new axes {new_axes} with new data type {new_data_type}, and current "
                f"axes {self.axes} with current data type {self.data_type}."
                f"{additional_msg}"
            )

        # normalize new_channels parameter to lift ambiguity around `None`
        #   - If None, keep previous parameter
        #   - If "all", select all channels (None value internally)
        if new_channels is None:
            new_channels = self.channels
        elif new_channels == "all":
            new_channels = None  # all channels

        # switching channels
        _validate_channel_conversion(
            self.axes,
            self.channels,
            new_axes or self.axes,  # if new_axes is None, we keep the same axes
            new_channels,  # new_channel has already been updated to the correct value
        )

        # apply default values
        patching_strategy: PatchingConfig
        if new_mode == Mode.PREDICTING:
            if new_patch_size is None:
                patching_strategy = WholePatchingConfig()
            else:
                if overlap_size is None:
                    raise ValueError(
                        "`overlap_size` parameter must be specified when switching to "
                        "'predicting' mode with a `new_patch_size`."
                    )
                patching_strategy = TiledPatchingConfig(
                    patch_size=list(new_patch_size), overlaps=list(overlap_size)
                )
        else:  # validating
            # to satisfy mypy, since self.mode=="training", patching has patch_size
            assert not isinstance(self.patching, WholePatchingConfig)

            patching_strategy = FixedRandomPatchingConfig(
                patch_size=(
                    list(new_patch_size)
                    if new_patch_size is not None
                    else self.patching.patch_size
                ),
            )

        # create new config
        model_dict = self.model_dump()
        model_dict.update(
            {
                "mode": new_mode,
                "patching": patching_strategy,
                "batch_size": new_batch_size or self.batch_size,
                "data_type": new_data_type or self.data_type,
                "axes": new_axes or self.axes,
                "channels": new_channels,
                "in_memory": (
                    new_in_memory if new_in_memory is not None else self.in_memory
                ),
                "val_dataloader_params": (
                    new_dataloader_params
                    if new_mode == Mode.VALIDATING and new_dataloader_params is not None
                    else self.val_dataloader_params
                ),
                "pred_dataloader_params": (
                    new_dataloader_params
                    if new_mode == Mode.PREDICTING and new_dataloader_params is not None
                    else self.pred_dataloader_params
                ),
                "patch_filter": None,
                "mask_filter": None,
            }
        )

        # remove patch filter when switching to validation or prediction
        del model_dict["patch_filter"]
        del model_dict["mask_filter"]

        return DataConfig(**model_dict)

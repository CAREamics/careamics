"""Data configuration."""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Sequence
from pprint import pformat
from typing import Annotated, Any, Literal, Self, Union
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    field_validator,
    model_validator,
)

from careamics.utils import BaseEnum

from ..transformations import XYFlipConfig, XYRandomRotate90Config
from ..validators import check_axes_validity, check_czi_axes_validity
from .patch_filter import (
    MaskFilterConfig,
    MaxFilterConfig,
    MeanSTDFilterConfig,
    ShannonFilterConfig,
)
from .patching_strategies import (
    FixedRandomPatchingConfig,
    RandomPatchingConfig,
    TiledPatchingConfig,
    WholePatchingConfig,
)

# TODO: Validate the specific sizes of tiles and overlaps given UNet constraints
#   - needs to be done in the Configuration
#   - patches and overlaps sizes must also be checked against dimensionality
#   - Should we have a UNet and a LVAE NGDataConfig subclass with specific validations?

# TODO: is 3D updated anywhere in the code in CAREamist/downstream?
#       - this will be important when swapping the data config in Configuration
#       - `set_3D` currently not implemented here

# TODO: this module is very long, can we split the validation somewhere else and
#       leverage Pydantic to add validation directly to the declaration of each field?


def generate_random_seed() -> int:
    """Generate a random seed for reproducibility.

    Returns
    -------
    int
        A random integer between 1 and 2^31 - 1.
    """
    return random.randint(1, 2**31 - 1)


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


Float = Annotated[float, PlainSerializer(np_float_to_scientific_str, return_type=str)]
"""Annotated float type, used to serialize floats to strings."""

PatchingConfig = Union[
    FixedRandomPatchingConfig,
    RandomPatchingConfig,
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

CoordFilterConfig = Union[MaskFilterConfig]  # add more here as needed
"""Coordinate filter type."""


class Mode(str, BaseEnum):
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


class NGDataConfig(BaseModel):
    """Next-Generation Dataset configuration.

    NGDataConfig are used for both training and prediction, with the patching strategy
    determining how the data is processed. Note that `random` is the only patching
    strategy compatible with training, while `tiled` and `whole` are only used for
    prediction.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer instead `set_means_and_stds` to set both at once. Means and stds are expected
    to be lists of floats, one for each channel. For supervised tasks, the mean and std
    of the target could be different from the input data.

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

    patching: PatchingConfig = Field(..., discriminator="name")
    """Patching strategy to use. Note that `random` is the only supported strategy for
    training, while `tiled` and `whole` are only used for prediction."""

    # Optional fields
    batch_size: int = Field(default=1, ge=1, validate_default=True)
    """Batch size for training."""

    in_memory: bool = Field(default_factory=default_in_memory, validate_default=True)
    """Whether to load all data into memory. This is only supported for 'array',
    'tiff' and 'custom' data types. Must be `True` for `array`. If `None`, defaults to
    `True` for 'array', 'tiff' and `custom`, and `False` for 'zarr' and 'czi' data
    types."""

    channels: Sequence[int] | None = Field(default=None)
    """Channels to use from the data. If `None`, all channels are used."""

    patch_filter: PatchFilterConfig | None = Field(default=None, discriminator="name")
    """Patch filter to apply when using random patching. Only available if
    mode is `training`."""

    coord_filter: CoordFilterConfig | None = Field(default=None, discriminator="name")
    """Coordinate filter to apply when using random patching. Only available if
    mode is `training`."""

    patch_filter_patience: int = Field(default=5, ge=1)
    """Number of consecutive patches not passing the filter before accepting the next
    patch."""

    image_means: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Means of the data across channels, used for normalization."""

    image_stds: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Standard deviations of the data across channels, used for normalization."""

    target_means: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Means of the target data across channels, used for normalization."""

    target_stds: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Standard deviations of the target data across channels, used for
    normalization."""

    transforms: Sequence[Union[XYFlipConfig, XYRandomRotate90Config]] = Field(
        default=(
            XYFlipConfig(),
            XYRandomRotate90Config(),
        ),
        validate_default=True,
    )
    """List of transformations to apply to the data, available transforms are defined
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

    seed: int | None = Field(default_factory=generate_random_seed, gt=0)
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
        # Additional validation for CZI files
        if info.data["data_type"] == "czi":
            if not check_czi_axes_validity(axes):
                raise ValueError(
                    f"Provided axes '{axes}' are not valid. Axes must be in the "
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

        if in_memory and data_type not in ("array", "tiff", "custom"):
            raise ValueError(
                f"`in_memory` can only be True for 'array', 'tiff' and 'custom' "
                f"data types, got '{data_type}'. In memory loading of zarr and czi "
                f"data types is not currently not implemented."
            )

        if not in_memory and data_type == "array":
            raise ValueError(
                "`in_memory` must be True for 'array' data type, got False."
            )

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
                    "Channels were specified but 'C' is not present in the axes."
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

        - If mode is `training`, patching strategy must be `random`.
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
            if patching.name != "random":
                raise ValueError(
                    f"Patching strategy '{patching.name}' is not compatible with "
                    f"mode '{mode.value}'. Use 'random' for training."
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

    @field_validator("patch_filter", "coord_filter")
    @classmethod
    def validate_filters_against_mode(
        cls,
        filter_obj: PatchFilterConfig | CoordFilterConfig | None,
        info: ValidationInfo,
    ) -> PatchFilterConfig | CoordFilterConfig | None:
        """
        Validate that the filters are only used during training.

        Parameters
        ----------
        filter_obj : PatchFilters or CoordFilters or None
            Filter to validate.
        info : ValidationInfo
            Validation information.

        Returns
        -------
        PatchFilters or CoordFilters or None
            Validated filter.

        Raises
        ------
        ValueError
            If a filter is used in a mode other than training.
        """
        mode = info.data["mode"]
        if filter_obj is not None and mode != Mode.TRAINING:
            raise ValueError(
                f"Filter '{filter_obj.name}' can only be used in 'training' mode, "
                f"got mode '{mode.value}'."
            )
        return filter_obj

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
                "Value for 'shuffle' was not included in the `train_dataloader_params`."
            )
        elif ("shuffle" in train_dataloader_params) and (
            not train_dataloader_params["shuffle"]
        ):
            warn(
                "Dataloader parameters include `shuffle=False`, this will be passed to "
                "the training dataloader and may lead to lower quality results.",
                stacklevel=1,
            )
        return train_dataloader_params

    @model_validator(mode="after")
    def std_only_with_mean(self: Self) -> Self:
        """
        Check that mean and std are either both None, or both specified.

        Returns
        -------
        Self
            Validated data model.

        Raises
        ------
        ValueError
            If std is not None and mean is None.
        """
        # check that mean and std are either both None, or both specified
        if (self.image_means and not self.image_stds) or (
            self.image_stds and not self.image_means
        ):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        elif (self.image_means is not None and self.image_stds is not None) and (
            len(self.image_means) != len(self.image_stds)
        ):
            raise ValueError("Mean and std must be specified for each input channel.")

        if (self.target_means and not self.target_stds) or (
            self.target_stds and not self.target_means
        ):
            raise ValueError(
                "Mean and std must be either both None, or both specified "
            )

        elif self.target_means is not None and self.target_stds is not None:
            if len(self.target_means) != len(self.target_stds):
                raise ValueError(
                    "Mean and std must be either both None, or both specified for each "
                    "target channel."
                )

        return self

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
    def propagate_seed_to_filters(self: Self) -> Self:
        """
        Propagate the main seed to patch and coordinate filters that support seeds.

        This ensures that all filters use the same seed for reproducibility,
        unless they already have a seed explicitly set.

        Returns
        -------
        Self
            Data model with propagated seeds.
        """
        if self.seed is not None:
            if self.patch_filter is not None:
                if (
                    hasattr(self.patch_filter, "seed")
                    and self.patch_filter.seed is None
                ):
                    self.patch_filter.seed = self.seed

            if self.coord_filter is not None:
                if (
                    hasattr(self.coord_filter, "seed")
                    and self.coord_filter.seed is None
                ):
                    self.coord_filter.seed = self.seed

        return self

    @model_validator(mode="after")
    def propagate_seed_to_transforms(self: Self) -> Self:
        """
        Propagate the main seed to all transforms that support seeds.

        This ensures that all transforms use the same seed for reproducibility,
        unless they already have a seed explicitly set.

        Returns
        -------
        Self
            Data model with propagated seeds.
        """
        if self.seed is not None:
            for transform in self.transforms:
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

    @field_validator("train_dataloader_params", mode="before")
    @classmethod
    def set_default_train_workers(
        cls, dataloader_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Set default num_workers for training dataloader if not provided.

        - If 'num_workers' is not set, it defaults to the number of available CPU cores.

        Parameters
        ----------
        dataloader_params : dict of {str: Any}
            The training dataloader parameters.

        Returns
        -------
        dict of {str: Any}
            The dataloader parameters with num_workers default applied.
        """
        if "num_workers" not in dataloader_params:
            # Use 0 workers during tests, otherwise use all available CPU cores
            if "pytest" in sys.modules:
                dataloader_params["num_workers"] = 0
            else:
                dataloader_params["num_workers"] = os.cpu_count()

        return dataloader_params

    @model_validator(mode="after")
    def set_val_workers_to_match_train(self: Self) -> Self:
        """
        Set validation dataloader num_workers to match training dataloader.

        If num_workers is not specified in val_dataloader_params, it will be set to the
        same value as train_dataloader_params["num_workers"].

        Returns
        -------
        Self
            Validated data model with synchronized num_workers.
        """
        if "num_workers" not in self.val_dataloader_params:
            self.val_dataloader_params["num_workers"] = self.train_dataloader_params[
                "num_workers"
            ]
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

    def _update(self, **kwargs: Any) -> None:
        """
        Update multiple arguments at once.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments to update.
        """
        self.__dict__.update(kwargs)
        self.__class__.model_validate(self.__dict__)

    def set_means_and_stds(
        self,
        image_means: Union[NDArray, tuple, list, None],
        image_stds: Union[NDArray, tuple, list, None],
        target_means: Union[NDArray, tuple, list, None] | None = None,
        target_stds: Union[NDArray, tuple, list, None] | None = None,
    ) -> None:
        """
        Set mean and standard deviation of the data across channels.

        This method should be used instead setting the fields directly, as it would
        otherwise trigger a validation error.

        Parameters
        ----------
        image_means : numpy.ndarray, tuple or list
            Mean values for normalization.
        image_stds : numpy.ndarray, tuple or list
            Standard deviation values for normalization.
        target_means : numpy.ndarray, tuple or list, optional
            Target mean values for normalization, by default ().
        target_stds : numpy.ndarray, tuple or list, optional
            Target standard deviation values for normalization, by default ().
        """
        # make sure we pass a list
        if image_means is not None:
            image_means = list(image_means)
        if image_stds is not None:
            image_stds = list(image_stds)
        if target_means is not None:
            target_means = list(target_means)
        if target_stds is not None:
            target_stds = list(target_stds)

        self._update(
            image_means=image_means,
            image_stds=image_stds,
            target_means=target_means,
            target_stds=target_stds,
        )

    def is_3D(self) -> bool:
        """
        Check if the data is 3D based on the axes.

        Either "Z" is in the axes and patching `patch_size` has 3 dimensions, or for CZI
        data, "Z" is in the axes or "T" is in the axes and patching `patch_size` has
        3 dimensions.

        This method is used during NGConfiguration validation to cross checks dimensions
        with the algorithm configuration.

        Returns
        -------
        bool
            True if the data is 3D, False otherwise.
        """
        if self.data_type == "czi":
            return "Z" in self.axes or "T" in self.axes
        else:
            return "Z" in self.axes

    # TODO: if switching from a state in which in_memory=True to an incompatible state
    # an error will be raised. Should that automatically be set to False instead?
    # TODO `channels=None` is ambigouous: all channels or same channels as in training?
    # TODO this method could be private and we could have public `to_validation_config`
    #   and `to_prediction_config` methods with appropriate parameters
    # TODO any use for switching to training mode?
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
    ) -> NGDataConfig:
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
        NGDataConfig
            New NGDataConfig with the updated mode and parameters.

        Raises
        ------
        ValueError
            If conversion to training mode is requested, or if incompatible changes
            are requested.
        """
        if new_mode == Mode.TRAINING:
            raise ValueError(
                "Converting to 'training' mode is not supported. Create a new "
                "NGDataConfig instead, for instance using "
                "`create_ng_data_configuration`."
            )
        if self.mode != Mode.TRAINING:
            raise ValueError(
                f"Converting from mode '{self.mode}' to '{new_mode}' is not supported. "
                f"Only conversion from 'training' mode is supported."
            )

        # sanity checks
        # switching spatial axes
        if new_axes is not None and ("Z" in new_axes) != (
            "Z" in self.axes
        ):  # switching 2D/3D
            raise ValueError("Cannot switch between 2D and 3D axes.")

        # normalize new_channels parameter to lift ambiguity around `None`
        #   - If None, keep previous parameter
        #   - If "all", select all channels (None value internally)
        if new_channels is None:
            new_channels = self.channels
        elif new_channels == "all":
            new_channels = None  # all channels

        # switching channels
        # if switching C axis:
        # - removing C: original channels can be `None`, singleton or multiple. New
        #   channels can be `None` if original were `None` or singleton, but not
        #   multiple.
        # - adding C: original channels can only be `None`. New channels can be `None`
        #   (but we warn users that they need to have a singleton C axis in the data),
        #   or singleton, but not multiple.
        adding_C_axis = (
            new_axes is not None and ("C" in new_axes) and ("C" not in self.axes)
        )
        removing_C_axis = (
            new_axes is not None and ("C" not in new_axes) and ("C" in self.axes)
        )
        prev_channels_not_singleton = self.channels is not None and (
            len(self.channels) != 1
        )

        if adding_C_axis:
            if new_channels is None:
                warn(
                    f"When switching to axes with 'C' (got {new_axes}) from axes "
                    f"{self.axes}, errors may be raised or degraded performances may be"
                    f" observed if the channel dimension in the data is not a singleton"
                    f" dimension. To select a specific channel, use the `new_channels` "
                    f"parameter (e.g. `new_channels=[1]`).",
                    stacklevel=1,
                )
            elif len(new_channels) != 1:
                raise ValueError(
                    f"When switching to axes with 'C' (got {new_axes}) from axes "
                    f"{self.axes}, a single channel only must be selected using the "
                    f"`new_channels` parameter (got {new_channels})."
                )
        elif removing_C_axis and prev_channels_not_singleton:
            raise ValueError(
                f"Cannot switch to axes without 'C' (got {new_axes}) from axes "
                f"{self.axes} when multiple channels were originally specified "
                f"({self.channels})."
            )

        # different number of channels
        if new_channels is not None and self.channels is not None:
            if len(new_channels) != len(self.channels):
                raise ValueError(
                    f"New channels length ({len(new_channels)}) does not match "
                    f"current channels length ({len(self.channels)})."
                )

        if self.channels is None and new_channels is not None:
            warn(
                f"Switching from all channels (`channels=None`) to specifying channels "
                f"{new_channels} may lead to errors or degraded performances if "
                f"{new_channels} are not all channels.",
                stacklevel=1,
            )  # Note that in the opposite case, self.channels is kept because
            # new_channels is None

        # apply default values
        patching_strategy: PatchingConfig
        if new_mode == Mode.PREDICTING:
            if new_patch_size is None:
                patching_strategy = WholePatchingConfig()
            else:
                if overlap_size is None:
                    raise ValueError(
                        "When switching to 'predicting' mode with 'tiled' patching, "
                        "the `overlap_size` parameter must be specified."
                    )
                patching_strategy = TiledPatchingConfig(
                    patch_size=list(new_patch_size), overlaps=list(overlap_size)
                )
        else:  # validating
            assert isinstance(self.patching, RandomPatchingConfig)  # for mypy

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
                "channels": new_channels if new_channels is not None else self.channels,
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
            }
        )

        # remove patch and coord filters when switching to validation or prediction
        del model_dict["patch_filter"]
        del model_dict["coord_filter"]

        return NGDataConfig(**model_dict)

    # def set_3D(self, axes: str, patch_size: list[int]) -> None:
    #     """
    #     Set 3D parameters.

    #     Parameters
    #     ----------
    #     axes : str
    #         Axes.
    #     patch_size : list of int
    #         Patch size.
    #     """
    #     self._update(axes=axes, patch_size=patch_size)

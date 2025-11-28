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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from ..transformations import XYFlipConfig, XYRandomRotate90Config
from ..validators import check_axes_validity
from .normalization_config import (
    MinMaxModel,
    NoNormModel,
    QuantileModel,
    StandardizeModel,
)
from .patch_filter import (
    MaskFilterConfig,
    MaxFilterConfig,
    MeanSTDFilterConfig,
    ShannonFilterConfig,
)
from .patching_strategies import (
    RandomPatchingConfig,
    TiledPatchingConfig,
    WholePatchingConfig,
)

# TODO: Validate the specific sizes of tiles and overlaps given UNet constraints
#   - needs to be done in the Configuration
#   - patches and overlaps sizes must also be checked against dimensionality

# TODO: is 3D updated anywhere in the code in CAREamist/downstream?
#       - this will be important when swapping the data config in Configuration
#       - `set_3D` currently not implemented here
# TODO: we can't tell that the patching strategy is correct
#       - or is the responsibility of the creator (e.g. convenience functions)


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

PatchingStrategies = Union[
    RandomPatchingConfig,
    # SequentialPatchingConfig, # not supported yet
    TiledPatchingConfig,
    WholePatchingConfig,
]
"""Patching strategies."""

NormalizationStrategies = Union[
    StandardizeModel,
    NoNormModel,
    QuantileModel,
    MinMaxModel,
]
"""Normalization strategies."""

PatchFilters = Union[
    MaxFilterConfig,
    MeanSTDFilterConfig,
    ShannonFilterConfig,
]
"""Patch filters."""

CoordFilters = Union[MaskFilterConfig]  # add more here as needed
"""Coordinate filters."""


class NGDataConfig(BaseModel):
    """Next-Generation Dataset configuration.

    NGDataConfig are used for both training and prediction, with the patching strategy
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
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"]
    """Type of input data."""

    axes: str
    """Axes of the data, as defined in SupportedAxes."""

    patching: PatchingStrategies = Field(..., discriminator="name")
    """Patching strategy to use. Note that `random` is the only supported strategy for
    training, while `tiled` and `whole` are only used for prediction."""

    normalization: NormalizationStrategies = Field(..., discriminator="name")
    """Normalization strategy to use."""

    # Optional fields
    batch_size: int = Field(default=1, ge=1, validate_default=True)
    """Batch size for training."""

    patch_filter: PatchFilters | None = Field(default=None, discriminator="name")
    """Patch filter to apply when using random patching. Only available during
    training."""

    coord_filter: CoordFilters | None = Field(default=None, discriminator="name")
    """Coordinate filter to apply when using random patching. Only available during
    training."""

    patch_filter_patience: int = Field(default=5, ge=1)
    """Number of consecutive patches not passing the filter before accepting the next
    patch."""

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
    def axes_valid(cls, axes: str) -> str:
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

        Returns
        -------
        str
            Validated axes.

        Raises
        ------
        ValueError
            If axes are not valid.
        """
        # Validate axes
        check_axes_validity(axes)

        return axes

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
        if "Z" in self.axes:
            if (
                hasattr(self.patching, "patch_size")
                and len(self.patching.patch_size) != 3
            ):
                raise ValueError(
                    f"`patch_size` in `patching` must have 3 dimensions if the data is"
                    f" 3D, got axes {self.axes})."
                )
        else:
            if (
                hasattr(self.patching, "patch_size")
                and len(self.patching.patch_size) != 2
            ):
                raise ValueError(
                    f"`patch_size` in `patching` must have 2 dimensions if the data is"
                    f" 3D, got axes {self.axes})."
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

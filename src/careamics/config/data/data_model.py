"""Data configuration."""

from __future__ import annotations

import os
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
    field_validator,
    model_validator,
)

from ..transformations import XYFlipModel, XYRandomRotate90Model
from ..validators import check_axes_validity, patch_size_ge_than_8_power_of_2


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


class DataConfig(BaseModel):
    """Data configuration.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer instead `set_mean_and_std` to set both at once. Means and stds are expected
    to be lists of floats, one for each channel. For supervised tasks, the mean and std
    of the target could be different from the input data.

    All supported transforms are defined in the SupportedTransform enum.

    Examples
    --------
    Minimum example:

    >>> data = DataConfig(
    ...     data_type="array", # defined in SupportedData
    ...     patch_size=[128, 128],
    ...     batch_size=4,
    ...     axes="YX"
    ... )

    To change the image_means and image_stds of the data:
    >>> data.set_means_and_stds(image_means=[214.3], image_stds=[84.5])

    One can pass also a list of transformations, by keyword, using the
    SupportedTransform value:
    >>> from careamics.config.support import SupportedTransform
    >>> data = DataConfig(
    ...     data_type="tiff",
    ...     patch_size=[128, 128],
    ...     batch_size=4,
    ...     axes="YX",
    ...     transforms=[
    ...         {
    ...             "name": "XYFlip",
    ...         }
    ...     ]
    ... )
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Dataset configuration
    data_type: Literal["array", "tiff", "czi", "custom"]
    """Type of input data, numpy.ndarray (array) or paths (tiff, czi, and custom), as
    defined in SupportedData."""

    axes: str
    """Axes of the data, as defined in SupportedAxes."""

    patch_size: Union[list[int]] = Field(..., min_length=2, max_length=3)
    """Patch size, as used during training."""

    batch_size: int = Field(default=1, ge=1, validate_default=True)
    """Batch size for training."""

    # Optional fields
    image_means: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Means of the data across channels, used for normalization."""

    image_stds: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Standard deviations of the data across channels, used for normalization."""

    target_means: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Means of the target data across channels, used for normalization."""

    target_stds: list[Float] | None = Field(default=None, min_length=0, max_length=32)
    """Standard deviations of the target data across channels, used for
    normalization."""

    transforms: Sequence[Union[XYFlipModel, XYRandomRotate90Model]] = Field(
        default=[
            XYFlipModel(),
            XYRandomRotate90Model(),
        ],
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

    val_dataloader_params: dict[str, Any] = Field(default={}, validate_default=True)
    """Dictionary of PyTorch validation dataloader parameters."""

    @field_validator("patch_size")
    @classmethod
    def all_elements_power_of_2_minimum_8(
        cls, patch_list: Union[list[int]]
    ) -> Union[list[int]]:
        """
        Validate patch size.

        Patch size must be powers of 2 and minimum 8.

        Parameters
        ----------
        patch_list : list of int
            Patch size.

        Returns
        -------
        list of int
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size is smaller than 8.
        ValueError
            If the patch size is not a power of 2.
        """
        patch_size_ge_than_8_power_of_2(patch_list)

        return patch_list

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
        Validate 2D/3D dimensions between axes, patch size and transforms.

        Returns
        -------
        Self
            Validated data model.

        Raises
        ------
        ValueError
            If the transforms are not valid.
        """
        if "Z" in self.axes:
            if len(self.patch_size) != 3:
                raise ValueError(
                    f"Patch size must have 3 dimensions if the data is 3D "
                    f"({self.axes})."
                )

        else:
            if len(self.patch_size) != 2:
                raise ValueError(
                    f"Patch size must have 3 dimensions if the data is 3D "
                    f"({self.axes})."
                )

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
        self._update(axes=axes, patch_size=patch_size)

"""Data configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pprint import pformat
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
)

from ..transformations import XYFlipModel, XYRandomRotate90Model
from ..validators import check_axes_validity


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
    """General data configuration.

    This model is used to define the data configuration parameters but the actual
    implementation used throughout CAREamics are the `TrainingDataConfig` and the
    `PredictionDataConfig`.

    To be valid, axes must abide the following constraints:
    - be a combination of 'STCZYX'
    - not contain duplicates
    - contain at least 2 contiguous axes: X and Y
    - contain at most 4 axes
    - not contain both S and T axes

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
    data_type: Literal["array", "tiff", "custom"]
    """Type of input data, numpy.ndarray (array) or paths (tiff and custom), as defined
    in SupportedData."""

    axes: str
    """Axes of the data, as defined in SupportedAxes."""

    patch_size: Optional[Union[list[int]]] = Field(
        default=None, min_length=2, max_length=3
    )
    """Patch size."""

    batch_size: int = Field(default=1, ge=1, validate_default=True)
    """Batch size."""

    # Optional fields
    image_means: Optional[list[Float]] = Field(
        default=None, min_length=0, max_length=32
    )
    """Means of the data across channels, used for normalization."""

    image_stds: Optional[list[Float]] = Field(default=None, min_length=0, max_length=32)
    """Standard deviations of the data across channels, used for normalization."""

    target_means: Optional[list[Float]] = Field(
        default=None, min_length=0, max_length=32
    )
    """Means of the target data across channels, used for normalization during
    training."""

    target_stds: Optional[list[Float]] = Field(
        default=None, min_length=0, max_length=32
    )
    """Standard deviations of the target data across channels, used for
    normalization during training."""

    transforms: Sequence[Union[XYFlipModel, XYRandomRotate90Model]] = Field(
        default=[],
        validate_default=True,
    )
    """List of transformations to apply to the data, available transforms are defined
    in SupportedTransform."""

    train_dataloader_params: dict[str, Any] = Field(default={}, validate_default=True)
    """Dictionary of PyTorch training dataloader parameters."""

    val_dataloader_params: Optional[dict[str, Any]] = Field(default={})
    """Dictionary of PyTorch validation dataloader parameters."""

    random_seed: Optional[int] = Field(default=None, ge=0)
    """Random seed for reproducibility."""

    patch_overlaps: Optional[list[int]] = Field(
        default=None, min_length=2, max_length=3
    )
    """Overlap between patches, only used during prediction."""

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

    # TODO will this be used?
    def set_means_and_stds(
        self,
        image_means: Union[NDArray, tuple, list, None],
        image_stds: Union[NDArray, tuple, list, None],
        target_means: Optional[Union[NDArray, tuple, list, None]] = None,
        target_stds: Optional[Union[NDArray, tuple, list, None]] = None,
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

    # TODO usunused, remove?
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

    def is_tiled(self) -> bool:
        """
        Check if the data should be tiled.

        Data should be tiled if both `patch_size` and `patch_overlaps` are not None.

        Returns
        -------
        bool
            True if the data should be tiled, False otherwise.
        """
        return self.patch_size is not None and self.patch_overlaps is not None

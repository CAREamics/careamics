"""Data configuration."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .transform import TransformModel
from ..utils import check_axes_validity


# TODO: how to switch off the transforms? empty list []?
class Data(BaseModel):
    """
    Data configuration.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer instead the following:
    >>> set_mean_and_std(mean, std)

    Attributes
    ----------
    in_memory : bool
        Whether to load the data in memory or not.
    data_format : SupportedExtension
        Extension of the data, without period.
    axes : str
        Axes of the data.
    mean: Optional[float]
        Expected data mean.
    std: Optional[float]
        Expected data standard deviation.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Mandatory fields
    data_type: Literal["Array", "Tiff", "Zarr", "Custom"]
    patch_size: List[int] = Field(..., min_length=2, max_length=3)

    axes: str

    # Optional fields
    mean: Optional[float] = Field(default=None, ge=0)
    std: Optional[float] = Field(default=None, gt=0)
    transforms: Optional[List[TransformModel]] = None


    @field_validator("axes")
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

    @field_validator("patch_size")
    def all_elements_non_zero_even(cls, patch_list: List[int]) -> List[int]:
        """
        Validate patch size.

        Patch size must be non-zero, positive and even.

        Parameters
        ----------
        patch_list : List[int]
            Patch size.

        Returns
        -------
        List[int]
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size is 0.
        ValueError
            If the patch size is not even.
        """
        for dim in patch_list:
            if dim < 1:
                raise ValueError(f"Patch size must be non-zero positive (got {dim}).")

            if dim % 2 != 0:
                raise ValueError(f"Patch size must be even (got {dim}).")

        return patch_list


    def set_mean_and_std(self, mean: float, std: float) -> None:
        """
        Set mean and standard deviation of the data.

        This method is preferred to setting the fields directly, as it ensures that the
        mean is set first, then the std; thus avoiding a validation error to be thrown.

        Parameters
        ----------
        mean : float
            Mean of the data.
        std : float
            Standard deviation of the data.
        """
        self.mean = mean
        self.std = std

    # @model_validator(mode="after")
    # @classmethod
    # def validate_dataset_to_be_used(cls, data: Data) -> Any:
    #     """Validate that in_memory dataset is used correctly.

    #     Parameters
    #     ----------
    #     data : Configuration
    #         Configuration to validate.

    #     Raises
    #     ------
    #     ValueError
    #         If in_memory dataset is used with Zarr storage.
    #         If axes are not valid.
    #     """
    #     # TODO: why not? What if it is a small Zarr...
    #     if data.in_memory and data.data_format == SupportedExtension.ZARR:
    #         raise ValueError("Zarr storage can't be used with in_memory dataset.")

    #     return data

    # TODO is there a more elegant way? We could have an optional pydantic model with both specified!!
    @model_validator(mode="after")
    def std_only_with_mean(cls, data_model: Data) -> Data:
        """
        Check that mean and std are either both None, or both specified.

        If we enforce both None or both specified, we cannot set the values one by one
        due to the ConfDict enforcing the validation on assignment. Therefore, we check
        only when the std is not None and the mean is None.

        Parameters
        ----------
        data_model : Data
            Data model.

        Returns
        -------
        Data
            Validated data model.

        Raises
        ------
        ValueError
            If std is not None and mean is None.
        """
        if data_model.std is not None and data_model.mean is None:
            raise ValueError("Cannot have std non None if mean is None.")

        return data_model

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..utils import check_axes_validity


class SupportedExtensions(str, Enum):
    """Supported extensions for input data.

    Currently supported:
        - tif/tiff: .tiff files.
        - npy: numpy files.
    """

    TIFF = "tiff"
    TIF = "tif"
    NPY = "npy"

    @classmethod
    def _missing_(cls, value: object) -> str:
        """Override default behaviour for missing values.

        This method is called when `value` is not found in the enum values. It convert
        `value` to lowercase, remove "." if it is the first character and try to match
        it with enum values.
        """
        if isinstance(value, str):
            lower_value = value.lower()

            if lower_value.startswith("."):
                lower_value = lower_value[1:]

            # attempt to match lowercase value with enum values
            for member in cls:
                if member.value == lower_value:
                    return member

        # still missing
        return super()._missing_(value)


class Data(BaseModel):
    """Data configuration.

    The data paths are individually optional, however, at least one of training +
    validation or prediction must be specified.

    The optional paths to the training, validation and prediction data should point to
    the parent folder of the images.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer using the `set_mean_and_std` method instead.

    Attributes
    ----------
    data_format : SupportedExtensions
        Extensions of the data.
    axes : str
        Axes of the data.
    mean: Optional[float]
       Expected data mean
    std: Optional[float]
       Expected data std
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory fields
    data_format: SupportedExtensions
    axes: str

    # Optional fields
    mean: Optional[float] = Field(default=None, ge=0)
    std: Optional[float] = Field(default=None, gt=0)

    def set_mean_and_std(self, mean: float, std: float) -> None:
        """Set mean and std of the data.

        This method is preferred to setting the field directly, as it ensures that the
        mean is set first, then the std; thus avoiding a validation error to be thrown.
        """
        self.mean = mean
        self.std = std

    @field_validator("axes")
    def valid_axes(cls, axes: str) -> str:
        """Validate axes.

        Axes must be a subset of STZYX, must contain YX, be in the right order
        and not contain both S and T.

        Parameters
        ----------
        axes : str
            Axes of the training data
        values : dict
            Dictionary of other parameter values

        Returns
        -------
        str
            Axes of the training data

        Raises
        ------
        ValueError
            If axes are not valid
        """
        # validate axes
        check_axes_validity(axes)

        return axes

    @model_validator(mode="after")
    def std_only_with_mean(cls, data_model: Data) -> Data:
        """Check that mean and std are either both None, or both specified.

        If we enforce both None or both specified, we cannot set the values one by one
        due to the ConfDict enforcing the validation on assignment. Therefore, we check
        only when the std is not None and the mean is None.
        """
        if data_model.std is not None and data_model.mean is None:
            raise ValueError("Cannot have std non None if mean is None.")

        return data_model

    def model_dump(self, *args: List, **kwargs: Dict) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        return super().model_dump(exclude_none=True)

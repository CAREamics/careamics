from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldValidationInfo,
    field_validator,
    model_validator,
)

from ..utils import check_axes_validity

# TODO this creates a circular import when instantiating the engine
# engine -> config -> evaluation -> data -> dataloader_utils
# then are_axes_valid are imported again in the engine.
from .config_filter import paths_to_str


class SupportedExtensions(str, Enum):
    """Supported extensions for input data.

    Currently supported:
        - tif/tiff: .tiff files.
        - npy: numpy files.
    """

    TIFF = "tiff"
    TIF = "tif"
    NPY = "npy"  # TODO remove numpy after we reupload the dataset as zarr
    ZARR = "zarr"

    @classmethod
    def _missing_(cls, value: object):
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
    training_path : Optional[Union[Path, str]]
        Path to the training data.
    validation_path : Optional[Union[Path, str]]
        Path to the validation data.
    prediction_path : Optional[Union[Path, str]]
        Path to the prediction data.
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
    training_path: Optional[Union[Path, str]] = None
    validation_path: Optional[Union[Path, str]] = None
    prediction_path: Optional[Union[Path, str]] = None

    mean: Optional[float] = Field(default=None, ge=0)
    std: Optional[float] = Field(default=None, gt=0)

    @field_validator("training_path", "validation_path", "prediction_path")
    def path_contains_images(cls, path_value: str, values: FieldValidationInfo) -> Path:
        """Validate folder path.

        Check that files with the correct extension can be found in the folder.
        """
        path = Path(path_value)

        # check that the path exists
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        elif not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        # check that the path contains files with the correct extension
        if "data_format" in values.data:
            ext = values.data["data_format"]

            if len(list(path.glob(f"*.{ext}"))) == 0:
                raise ValueError(f"No files with extension {ext} found in {path}.")
        else:
            raise ValueError(
                "Cannot check path validity without extension, make sure it has been "
                "correctly specified."
            )

        return path

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
    def at_least_one_path_valid(cls, data_model: Data) -> Data:
        """Validate that at least one of training or prediction paths is specified.

        Parameters
        ----------
        data_model : Data
            Data model to validate

        Returns
        -------
        Data
            Validated model

        Raises
        ------
        ValueError
            If neither training or prediction paths are specified
        """
        if data_model.training_path is None and data_model.prediction_path is None:
            raise ValueError(
                "At least one of training or prediction paths must be specified."
            )

        return data_model

    @model_validator(mode="after")
    def both_training_and_validation(cls, data_model: Data) -> Data:
        """Validate that both training and validation paths are specified.

        Parameters
        ----------
        data_model : Data
            Data model to validate

        Returns
        -------
        Data
            Validated model

        Raises
        ------
        ValueError
            If one of training or validation paths is specified and not the other
        """
        if (data_model.training_path is None) != (data_model.validation_path is None):
            raise ValueError("Both training and validation paths must be specified.")

        return data_model

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

    def model_dump(self, *args, **kwargs) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - replace Path by str
            - remove optional values if they have the default value

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        dictionary = super().model_dump(exclude_none=True)

        # replace Paths by str
        return paths_to_str(dictionary)

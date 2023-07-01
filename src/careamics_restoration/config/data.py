from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, FieldValidationInfo, field_validator

# TODO this creates a circular import when instantiating the engine
# engine -> config -> evaluation -> data -> dataloader_utils
# then are_axes_valid are imported again in the engine.
from ..utils import are_axes_valid


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
    def _missing_(cls, value: str):
        """Override default behaviour for missing values.

        This method is called when `value` is not found in the enum values. It
        convert `value` to lowercase and try to match it with enum values.
        """
        lower_value = value.lower()

        # attempt to match lowercase value with enum values
        for member in cls:
            if member.value == lower_value:
                return member

        # attempt to remove a starting "."
        if lower_value.startswith("."):
            lower_value = lower_value[1:]
            for member in cls:
                if member.value == lower_value:
                    return member

        # still missing
        return super()._missing_(value)


class Data(BaseModel):
    """Data configuration.

    The optional paths to the training, validation and prediction data should point to
    the data parent folder.

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
    """

    # Pydantic class configuration
    model_config = ConfigDict(use_enum_values=True)

    # Mandatory fields
    data_format: SupportedExtensions
    axes: str

    # Optional fields
    training_path: Optional[Union[Path, str]] = None
    validation_path: Optional[Union[Path, str]] = None
    prediction_path: Optional[Union[Path, str]] = None

    @field_validator("training_path", "validation_path", "prediction_path")
    def check_path(cls, path_value: str, values: FieldValidationInfo) -> Path:
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
            raise ValueError("Cannot check path validity without extension.")

        return path

    @field_validator("axes")
    def validate_axes(cls, axes: str) -> str:
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
        are_axes_valid(axes)

        return axes

    def dict(self, *args, **kwargs) -> dict:
        """Override dict method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - replace Path by str
        """
        dictionary = super().dict(exclude_none=True)

        # replace Path by str
        dictionary["folder_path"] = str(dictionary["folder_path"])

        return dictionary

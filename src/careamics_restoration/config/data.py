from enum import Enum
from pathlib import Path
from typing import Union

from pydantic import BaseModel, validator

# TODO this creates a circular import when instantiating the engine
# engine -> config -> evaluation -> data -> dataloader_utils
# then are_axes_valid are imported again in the engine.
from ..utils import are_axes_valid


class SupportedExtensions(str, Enum):
    """Supported extensions for input data.

    Currently supported:
        - tiff/tiff: tiff files.
        - npy: numpy files.
    """

    TIFF = "tiff"
    TIF = "tif"
    NPY = "npy"

    @classmethod
    def _missing_(cls, value):
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

    Attributes
    ----------
    folder_path : Union[Path, str]
        Path to the data, .
    data_format : SupportedExtensions
        Extensions of the data.
    axes : str
        Axes of the training data.
    """

    # Mandatory fields
    data_format: SupportedExtensions
    folder_path: Union[Path, str]
    axes: str

    @validator("path")
    def check_path(cls, path_value: str, values: dict) -> Path:
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
        if "data_format" in values:
            ext = values["data_format"]

            if len(list(path.glob(f"*.{ext}"))) == 0:
                raise ValueError(f"No files with extension {ext} found in {path}.")
        else:
            raise ValueError("Cannot check path validity without extension.")

        return path

    @validator("axes")
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

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable


class TrainingData(Data):
    @validator("path")
    def check_path(cls, path_value: str, values: dict) -> Path:
        """Validate folder path for training data.

        For training data, the folder path can contain training and validation
        sub-folder.
        """
        try:
            # check if it contains values with the correct extension
            path = super().check_path(path_value, values)
        except ValueError:
            # if it doesn't, we check for training and validation sub-folders.
            path = Path(path_value)
            train_path = path / "training"

            if not train_path.exists():
                raise ValueError(f"Path {path} does not contain a training sub-folder.")
            else:
                # check that the path contains files with the correct extension
                if "data_format" in values:
                    ext = values["data_format"]

                    if len(list(train_path.glob(f"*.{ext}"))) == 0:
                        raise ValueError(
                            f"No files with extension {ext} found in {path}."
                        )
                else:
                    raise ValueError("Cannot check path validity without extension.")

            return path

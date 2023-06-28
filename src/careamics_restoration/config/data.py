from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

# TODO this creates a circular import when instantiating the engine
# engine -> config -> evaluation -> data -> dataloader_utils
# then are_axes_valid are imported again in the engine.
from ..utils import are_axes_valid


class SupportedExtension(str, Enum):
    """Supported extensions for input data."""

    TIFF = "tiff"
    TIF = "tif"
    NPY = "npy"

    @classmethod
    def _missing_(cls, value):
        """Override default behaviour for missing values.

        Convert value to lowercase and try to match it with enum values.
        """
        lower_value = value.lower()

        # attempt to match lowercase value with enum values
        for member in cls:
            if member.value == lower_value:
                return member

        return super()._missing_(value)


class ExtractionStrategy(str, Enum):
    """Extraction strategy for training patches."""

    SEQUENTIAL = "sequential"
    RANDOM = "random"
    PREDICT = "predict"


class Data(BaseModel):
    """Data configuration.

    Attributes
    ----------
    path: Path
        Path to the folder containing the training data or to a specific file (
        extension must match `ext`)
    axes: str
        Axes of the training data
    ext: SupportedExtension
        File type of the training data
    extraction_strategy: ExtractionStrategy
        Extraction strategy for training patches
    batch_size: int
        Batch size for training
    patch_size: List[int]
        Patch size for training, defines spatial dimensionality (2D vs 3D)
    num_files: Optional[int]
        Number of files to use for training (optional)
    num_patches: Optional[int]
        Number of patches to use for training (optional)
    num_workers: Optional[int]
        Number of workers for training (optional)
    """

    # optional with default values (included in yml)
    ext: SupportedExtension = SupportedExtension.TIF
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.SEQUENTIAL

    batch_size: int = Field(default=1, ge=1)

    path: Path
    patch_size: Optional[List[int]] = Field(..., min_items=2, max_items=3)
    axes: str

    # optional with None default values (not included in yml if not defined)
    num_files: Optional[int] = Field(default=None, ge=1)  # TODO why is this needed?
    num_patches: Optional[int] = Field(None, ge=1)
    num_workers: Field(default=0, ge=0, le=8) 

    # TODO how to make parameters mutually exclusive (which one???)

    @validator("path")
    def validate_path(cls, v: Union[Path, str], values: dict) -> Path:
        """Validate path to training data.

        Parameters
        ----------
        v : Union[Path, str]
            Path to training data
        values : dict
            Dictionary of other parameter values

        Returns
        -------
        Path
            Path to training data

        Raises
        ------
        ValueError
            If path does not exist, or if it has the wrong extension
        """
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        elif not path.is_dir():
            if "ext" in values:
                if path.suffix != f".{values['ext']}":
                    raise ValueError(
                        f"Path {path} does not have the expected extension"
                        f" {values['ext']}"
                    )
            else:
                raise ValueError("Cannot check path validity without extension.")

        return path

    @validator("patch_size")
    def validate_patch_size(cls, patch_size: List[int]) -> List[int]:
        """Validate patch size.

        For each entry:
        - check if power of 2
        - check if at minimum 8

        Parameters
        ----------
        patch_size : List[int]
            Patch size for training

        Returns
        -------
        List[int]
            Patch size for training

        Raises
        ------
        ValueError
            If patch size is not a power of 2 or at minimum 8
        """
        if patch_size is not None:
            for p in patch_size:
                # check if power of 2 and divisible by 8
                if not (p & (p - 1) == 0) or p < 8:
                    raise ValueError(
                        f"Patch size {p} is not a power of 2 or at minimum 8"
                    )
        return patch_size

    @validator("axes")
    def validate_axes(cls, axes: str, values: dict) -> str:
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

        # check if comaptible with patch size
        if "patch_size" in values:
            patch_size = values["patch_size"]

            # hard constraint
            if patch_size is not None:
                if len(axes) < len(patch_size):
                    raise ValueError(
                        f"Number of axes ({len(axes)}) cannot be smaller than patch"
                        f"size ({patch_size}) do not match."
                    )

                if len(patch_size) == 3 and "Z" not in axes:
                    raise ValueError(f"Missing Z axes in {axes=}.")
                elif len(patch_size) == 2 and "Z" in axes:
                    raise ValueError(f"Z axes in {axes=}, but patch size is 2D.")
        else:
            raise ValueError("Cannot check axes validity without patch size.")

        return axes

    @validator("num_files")
    def validate_num_files(
        cls, num_files: Optional[int], values: dict
    ) -> Optional[int]:
        if num_files is not None:
            if "ext" in values and "path" in values:
                file_list = list(values["path"].glob("*." + values["ext"]))
                if num_files != len(file_list):
                    raise ValueError(
                        f"Number of files ({len(file_list)}) found does not "
                        f"match num_files ({num_files})."
                    )
            else:
                raise ValueError(
                    "Cannot check num_files validity without extension and path."
                )

        return num_files

    def dict(self, *args, **kwargs) -> dict:
        """Override dict method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - replace Path by str
        """
        dictionary = super().dict(exclude_none=True)

        # replace Path by str
        dictionary["path"] = str(dictionary["path"])

        return dictionary

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable

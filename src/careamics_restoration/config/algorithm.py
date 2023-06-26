from enum import Enum
from pathlib import Path
from typing import Optional, Union, List

from pydantic import BaseModel, Field, validator


# python 3.11: https://docs.python.org/3/library/enum.html
class LossName(str, Enum):
    """Class representing an accepted loss function."""

    n2v = "n2v"
    pn2v = "pn2v"


class ModelName(str, Enum):
    """Class representing an accepted model."""

    UNET = "UNET"


class PixelManipulator(str, Enum):
    N2V = "n2v"


class Algorithm(BaseModel):
    """Algorithm configuration model.

    Attributes
    ----------
    loss : List[LossName]
        List of loss functions to be used for training (defined in n2v.losses)
    model : ModelName
        Model to be used for training (defined in n2v.models)
    depth : int
        Depth of the model (default: 3)
    conv_dims : int
        Dimensions of the convolution, 2D or 3D (default: 2)
    num_filter_base : int
        # TODO add description
    pixel_manipulation : PixelManipulator
        Pixel manipulation strategy, i.e. how are pixel masked during training
        (default: PixelManipulator.N2V)
    mask_pixel_percentage : float
        Percentage of pixels to be masked during training (default: 0.2%)
    trained_model : Optional[Path]
        Path to a trained model (default: None)
    """

    loss: Union[List[LossName], LossName]
    # workdir, mandatory field
    workdir: str
    # optional fields with default values (appearing in yml)
    # model
    model: ModelName = ModelName.UNET
    depth: int = Field(default=3, ge=2, le=5)
    conv_dims: int = Field(default=2, ge=2, le=3)
    num_filter_base: int = Field(default=96, ge=16, le=256)

    # pixel masking
    pixel_manipulation: PixelManipulator = PixelManipulator.N2V
    mask_pixel_percentage: float = Field(
        default=0.2, ge=0.1, le=5
    )  # TODO arbitrary maximum, justify otherwise 100.

    # optional fields that will not appear if not defined
    trained_model: Optional[str] = None

    @validator("workdir")
    def validate_workdir(cls, v: str, values, **kwargs) -> Path:
        """Validate trained_model.

        If trained_model is not None, it must be a valid path."""
        path = Path(v)
        if path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    @validator("trained_model")
    def validate_trained_model(cls, v: Union[Path, None], values, **kwargs) -> Path:
        """Validate trained_model.

        If trained_model is not None, it must be a valid path."""
        if v is not None:
            path = values["workdir"] / Path(v)
            if not path.exists():
                raise ValueError(f"Path to model does not exist (got {v}).")
            elif path.suffix != ".pth":
                raise ValueError(f"Path to model must be a .pth file (got {v}).")
            else:
                return path

        return None

    def dict(self) -> dict:
        """Override dict method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - replace Path by str
        """
        dictionary = super().dict(exclude_none=True)

        # replace Path by str
        dictionary["workdir"] = str(dictionary["workdir"])

        return dictionary

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable

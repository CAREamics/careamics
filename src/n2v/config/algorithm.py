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

    UNET = "UNet"


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
    pixel_manipulation : PixelManipulator
        Pixel manipulation strategy (default: PixelManipulator.N2V)
    num_masked_pixels : int
        Number of masked pixels (default: 128)
    trained_model : Optional[Path]
        Path to a trained model (default: None)
    """

    loss: Union[List[LossName], LossName]

    # optional fields with default values (appearing in yml)
    # model
    model: ModelName = ModelName.UNET
    depth: int = Field(default=3, ge=2, le=5)
    conv_dims: int = Field(default=2, ge=2, le=3)

    # pixel masking
    pixel_manipulation: PixelManipulator = PixelManipulator.N2V

    # TODO: should use percentage as in original N2V (absolute number makes no sense)
    num_masked_pixels: int = Field(default=128, ge=1, le=1024)

    # optional fields that will not appear if not defined
    trained_model: Optional[Path] = None

    @validator("trained_model")
    def validate_trained_model(cls, v: Union[Path, None], values, **kwargs) -> Path:
        """Validate trained_model.

        If trained_model is not None, it must be a valid path."""
        if v is not None:
            path = Path(v)
            if not v.exists():
                raise ValueError(f"Path to model does not exist (got {v}).")
            elif path.suffix != ".pth":
                raise ValueError(f"Path to model must be a .pth file (got {v}).")
            else:
                return path

        return None

    def dict(self, *args, **kwargs):
        """Return a dictionary representation of the model."""
        return super().dict(exclude_none=True)

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable

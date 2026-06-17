"""Pydantic model for the N2VManipulate transform."""

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from careamics.config.utils.random import generate_random_seed


def _odd_value(v: int) -> int:
    """
    Validate that the value is odd.

    Parameters
    ----------
    v : int
        Value to validate.

    Returns
    -------
    int
        The validated value.

    Raises
    ------
    ValueError
        If the value is even.
    """
    if v % 2 == 0:
        raise ValueError("Size must be an odd number.")
    return v


class StructMaskConfig(BaseModel):
    """Parameters of structN2V masks.

    Attributes
    ----------
    axes : Literal["horizontal", "vertical", "cross"]
        Axes along which to apply the mask, horizontal (0), vertical (1) or cross (2).
    span : int
        Span of the mask, must be odd.
    """

    axes: Literal["horizontal", "vertical", "cross"] = Field(default="horizontal")
    """Orientation of the structN2V mask."""

    span: Annotated[int, AfterValidator(_odd_value)] = Field(default=5, ge=3, le=15)
    """Size of the structN2V mask."""


class N2VManipulateConfig(BaseModel):
    """
    Configuration of the N2V manipulation transform.

    Attributes
    ----------
    name : Literal["N2VManipulate"]
        Name of the transformation.
    roi_size : int
        Size of the masking region, by default 11.
    masked_pixel_percentage : float
        Percentage of masked pixels, by default 0.2.
    strategy : Literal["uniform", "median"]
        Strategy pixel value replacement, by default "uniform".
    struct_mask : StructMaskConfig | None
        Parameters of the structN2V mask. If None, no structN2V mask is applied.
    seed : int
        Random seed for reproducibility.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    name: Literal["N2VManipulate"] = "N2VManipulate"

    roi_size: Annotated[int, AfterValidator(_odd_value)] = Field(
        default=11, ge=3, le=21
    )
    """Size of the region where the pixel manipulation is applied."""

    masked_pixel_percentage: float = Field(default=0.2, ge=0.05, le=10.0)
    """Percentage of masked pixels per image."""

    strategy: Literal["uniform", "median"] = Field(default="uniform")
    """Strategy for pixel value replacement."""

    struct_mask: StructMaskConfig | None = None
    """Parameters of the structN2V mask. If None, no structN2V mask is applied."""

    seed: int = Field(default_factory=generate_random_seed, gt=0)
    """Random seed for reproducibility. If not specified, a random seed is generated."""

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Return the model as a dictionary.

        Parameters
        ----------
        **kwargs
            Pydantic BaseMode model_dump method keyword arguments.

        Returns
        -------
        {str: Any}
            Dictionary representation of the model.
        """
        model_dict = super().model_dump(**kwargs)

        # remove the name field as it is not accepted by the augmentation class
        model_dict.pop("name")

        return model_dict

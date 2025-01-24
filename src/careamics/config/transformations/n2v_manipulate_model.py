"""Pydantic model for the N2VManipulate transform."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from .transform_model import TransformModel


# TODO should probably not be a TransformModel anymore, no reason for it
# `name` is used as a discriminator field in the transforms
class N2VManipulateModel(TransformModel):
    """
    Pydantic model used to represent N2V manipulation.

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
    struct_mask_axis : Literal["horizontal", "vertical", "none"]
        Axis of the structN2V mask, by default "none".
    struct_mask_span : int
        Span of the structN2V mask, by default 5.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["N2VManipulate"] = "N2VManipulate"

    roi_size: int = Field(default=11, ge=3, le=21)
    """Size of the region where the pixel manipulation is applied."""

    masked_pixel_percentage: float = Field(default=0.2, ge=0.05, le=10.0)
    """Percentage of masked pixels per image."""

    remove_center: bool = Field(default=True)  # TODO remove it
    """Exclude center pixel from average calculation."""  # TODO rephrase this

    strategy: Literal["uniform", "median"] = Field(default="uniform")
    """Strategy for pixel value replacement."""

    struct_mask_axis: Literal["horizontal", "vertical", "none"] = Field(default="none")
    """Orientation of the structN2V mask. Set to `\"non\"` to not apply StructN2V."""

    struct_mask_span: int = Field(default=5, ge=3, le=15)
    """Size of the structN2V mask."""

    @field_validator("roi_size", "struct_mask_span")
    @classmethod
    def odd_value(cls, v: int) -> int:
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

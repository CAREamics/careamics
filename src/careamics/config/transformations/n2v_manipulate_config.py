"""Pydantic model for the N2VManipulate transform."""

from typing import Literal, Optional

from pydantic import ConfigDict, Field, field_validator

from .transform_config import TransformConfig


# TODO should probably not be a TransformConfig anymore, no reason for it
# `name` is used as a discriminator field in the transforms
class N2VManipulateConfig(TransformConfig):
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
    data_channel_indices : Optional[list[int]]
        Specific channel indices to mask (e.g., [0, 3, 5]). If None, uses first n_data_channels.
    n_data_channels : int
        Number of data channels to mask (used when data_channel_indices is None).
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

    data_channel_indices: Optional[list[int]] = Field(
        default=None,
        description="Specific channel indices to mask (e.g., [0, 3, 5]). If None, uses first n_data_channels channels.",
    )
    """Specific channel indices to mask. If None, uses first n_data_channels."""

    n_data_channels: int = Field(
        default=1,
        ge=1,
        description="Number of data channels to mask starting from index 0 (used when data_channel_indices is None)",
    )
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

    @field_validator("data_channel_indices")
    @classmethod
    def validate_channel_indices(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        """
        Validate and sort data_channel_indices.

        Parameters
        ----------
        v : Optional[list[int]]
            Channel indices to validate.

        Returns
        -------
        Optional[list[int]]
            The validated and sorted channel indices.

        Raises
        ------
        ValueError
            If channel indices are invalid (negative, duplicates, or empty list).
        """
        if v is None:
            return v

        if len(v) == 0:
            raise ValueError("data_channel_indices cannot be an empty list. Use None instead.")

        if any(idx < 0 for idx in v):
            raise ValueError("Channel indices must be non-negative.")

        if len(v) != len(set(v)):
            raise ValueError("data_channel_indices cannot contain duplicates.")

        # Sort indices to ensure predictable mapping to output channels
        # Model output channel i will correspond to data_channel_indices[i]
        return sorted(v)

"""Pydantic model for the N2VManipulate transform."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from careamics.config.utils.random import generate_random_seed


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
    struct_mask_axis : Literal["horizontal", "vertical", "none"]
        Axis of the structN2V mask, by default "none".
    struct_mask_span : int
        Span of the structN2V mask, by default 5.
    seed : int
        Random seed for reproducibility.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    name: Literal["N2VManipulate"] = "N2VManipulate"

    roi_size: int = Field(default=11, ge=3, le=21)
    """Size of the region where the pixel manipulation is applied."""

    masked_pixel_percentage: float = Field(default=0.2, ge=0.05, le=10.0)
    """Percentage of masked pixels per image."""

    strategy: Literal["uniform", "median"] = Field(default="uniform")
    """Strategy for pixel value replacement."""

    struct_mask_axis: Literal["horizontal", "vertical", "none"] = Field(default="none")
    """Orientation of the structN2V mask. Set to `\"none\"` to not apply StructN2V."""

    struct_mask_span: int = Field(default=5, ge=3, le=15)
    """Size of the structN2V mask."""

    seed: int = Field(default_factory=generate_random_seed, gt=0)
    """Random seed for reproducibility. If not specified, a random seed is generated."""

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

"""Pydantic model for the Normalize transform."""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from .transform_model import TransformModel


class NormalizeModel(TransformModel):
    """
    Pydantic model used to represent Normalize transformation.

    The Normalize transform is a zero mean and unit variance transformation.

    Attributes
    ----------
    name : Literal["Normalize"]
        Name of the transformation.
    mean : float
        Mean value for normalization.
    std : float
        Standard deviation value for normalization.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["Normalize"] = "Normalize"
    image_means: list = Field(..., min_length=0, max_length=32)
    image_stds: list = Field(..., min_length=0, max_length=32)
    target_means: list | None = Field(default=None, min_length=0, max_length=32)
    target_stds: list | None = Field(default=None, min_length=0, max_length=32)

    @model_validator(mode="after")
    def validate_means_stds(self: Self) -> Self:
        """Validate that the means and stds have the same length.

        Returns
        -------
        Self
            The instance of the model.
        """
        if len(self.image_means) != len(self.image_stds):
            raise ValueError("The number of image means and stds must be the same.")

        if (self.target_means is None) != (self.target_stds is None):
            raise ValueError(
                "Both target means and stds must be provided together, or bot None."
            )

        if self.target_means is not None and self.target_stds is not None:
            if len(self.target_means) != len(self.target_stds):
                raise ValueError(
                    "The number of target means and stds must be the same."
                )

        return self

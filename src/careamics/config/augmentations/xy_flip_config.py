"""Pydantic model for the XYFlip augmentation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class XYFlipConfig(BaseModel):
    """
    Configuration for the XYFlip augmentation.

    Attributes
    ----------
    name : Literal["XYFlip"]
        Name of the transformation.
    p : float
        Probability of applying the transform, by default 0.5.
    seed : int | None
        Seed for the random number generator,  by default None.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    name: Literal["XYFlip"] = "XYFlip"
    flip_x: bool = Field(
        True,
        description="Whether to flip along the X axis.",
    )
    flip_y: bool = Field(
        True,
        description="Whether to flip along the Y axis.",
    )
    p: float = Field(
        0.5,
        description="Probability of applying the transform.",
        ge=0,
        le=1,
    )
    seed: int | None = None

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

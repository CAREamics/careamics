"""Pydantic model for the input-target swapping augmentation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SwapInputTargetConfig(BaseModel):
    """Configuration for randomly swapping input and target arrays."""

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["SwapInputTarget"] = "SwapInputTarget"
    """Name of the transformation."""

    p: float = Field(0.5, ge=0, le=1)
    """"Probability of swapping input and target."""

    seed: int | None = None
    """Seed."""

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Return the transform parameters without the discriminator name.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the base class model dump.

        Returns
        -------
        dict[str, Any]
            Transform parameters without `name`.
        """
        model_dict = super().model_dump(**kwargs)
        model_dict.pop("name")
        return model_dict

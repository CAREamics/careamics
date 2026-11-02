"""Pydantic model for the tilt-axis random 90 degree rotation augmentation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# TODO tilt axis random rotate can select rotation 0, but the normal augmentation
# cannot, so we have a discrepancy in default `p`
class TiltAxisRandomRotate90Config(BaseModel):
    """Configuration for rotation in the plane perpendicular to a tilt axis."""

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["TiltAxisRandomRotate90"] = "TiltAxisRandomRotate90"
    """Name of the transformation."""

    tilt_axis: Literal["Z", "Y", "X"]
    """Tilt axis, perpendicular to the plane of rotation."""

    p: float = Field(1.0, ge=0, le=1)
    "Probability of applying the transform."

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

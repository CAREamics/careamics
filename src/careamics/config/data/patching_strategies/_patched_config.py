"""Generic patching Pydantic model."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class _PatchedConfig(BaseModel):
    """Generic patching Pydantic model.

    This model is only used for inheritance and validation purposes.
    """

    model_config = ConfigDict(
        extra="ignore",  # default behaviour, make it explicit
    )

    name: str
    """The name of the patching strategy."""

    patch_size: Sequence[int] = Field(..., min_length=2, max_length=3)
    """The size of the patch in each spatial dimensions, each patch size must be a power
    of 2 and larger than 8."""

"""Generic patching Pydantic model."""

from collections.abc import Sequence
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field


def is_squared_in_yx(patch_size: Sequence[int]) -> bool:
    """Check if the patch size is squared in YX.

    Parameters
    ----------
    patch_size : Sequence[int]
        The size of the patch in each spatial dimension.

    Returns
    -------
    bool
        True if the patch size is squared in YX, False otherwise.
    """
    return patch_size[-1] == patch_size[-2]


class _PatchedConfig(BaseModel):
    """Generic patching Pydantic model.

    This model is only used for inheritance and validation purposes.
    """

    name: str
    """The name of the patching strategy."""

    patch_size: Annotated[Sequence[int], AfterValidator(is_squared_in_yx)] = Field(
        ..., min_length=2, max_length=3
    )
    """The size of the patch in each spatial dimensions. Must be squared in YX."""

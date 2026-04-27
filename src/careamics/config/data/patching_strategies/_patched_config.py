"""Generic patching Pydantic model."""

from collections.abc import Sequence
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field


def is_squared_in_yx(patch_size: Sequence[int]) -> Sequence[int]:
    """Validate that the patch size is squared in YX.

    Parameters
    ----------
    patch_size : Sequence[int]
        The size of the patch in each spatial dimension.

    Returns
    -------
    Sequence[int]
        The input patch size if it is squared in YX.
    """
    if patch_size[-1] != patch_size[-2]:
        raise ValueError(
            f"Patch size must be squared in YX, but got sizes {patch_size[-1]} and "
            f"{patch_size[-2]}."
        )

    return patch_size


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

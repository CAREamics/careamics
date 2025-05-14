"""Generic patching Pydantic model."""

from collections.abc import Sequence
from typing import Union

from pydantic import BaseModel, Field, field_validator

from careamics.config.validators import patch_size_ge_than_8_power_of_2


class PatchedModel(BaseModel):
    """Generic patching Pydantic model."""

    name: str
    """The name of the patching strategy."""

    patch_size: Union[Sequence[int]] = Field(..., min_length=2, max_length=3)
    """The size of the patch in each spatial dimensions, each patch size must be a power
    of 2 and larger than 8."""

    @field_validator("patch_size")
    @classmethod
    def all_elements_power_of_2_minimum_8(
        cls, patch_list: Union[Sequence[int]]
    ) -> Union[Sequence[int]]:
        """
        Validate patch size.

        Patch size must be powers of 2 and minimum 8.

        Parameters
        ----------
        patch_list : Sequence of int
            Patch size.

        Returns
        -------
        Sequence of int
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size is smaller than 8.
        ValueError
            If the patch size is not a power of 2.
        """
        patch_size_ge_than_8_power_of_2(patch_list)

        return patch_list

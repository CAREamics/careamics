"""Sliding-window tiled patching Pydantic model."""

from collections.abc import Callable, Sequence
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, ValidationInfo, field_validator

from ._overlapping_patched_config import _OverlappingPatchedConfig


def all_positive(
    strict: bool,
) -> Callable[[Sequence[int] | None], Sequence[int] | None]:
    """Return a Pydantic validator function to ensure integer sequences are positive.

    Parameters
    ----------
    strict : bool
        Whether the values should be strictly positive, i.e. `strict=True` means that
        zero is not included.

    Returns
    -------
    Callable[[Sequence[int] | None], Sequence[int] | None]
        Function to use as a Pydantic validator.
    """

    def wrapped_validator(value: Sequence[int] | None) -> Sequence[int] | None:
        """Validate that elements in a sequence are positive.

        Parameters
        ----------
        value : Sequence[int] | None
            A sequence of elements or None.

        Returns
        -------
        Sequence[int] | None
            A sequence of elements or None.

        Raises
        ------
        ValueError
            If any of the elements in the sequence are not positive.
        """
        if value is None:
            return value

        lower_inclusive = 1 if strict else 0
        if any(elem < lower_inclusive for elem in value):
            if strict:
                msg = "Each element of {value} must be strictly positive."
            else:
                msg = "Each element of {value} must be positive (zero is allowed)."
            raise ValueError(msg.format(value=value))
        return value

    return wrapped_validator


# TODO: rename `overlaps` to `halo`? NOTE: This is annoying in covert mode
#
class SwitiPatchingConfig(_OverlappingPatchedConfig):
    """Sliding-window inner-tiled (SWITi) patching Pydantic model.

    Attributes
    ----------
    name : "switi"
        The name of the patching strategy.
    patch_size : sequence of int
        The size of the patch in each spatial dimension, each patch size must be a power
        of 2 and larger than 8.
    overlaps : sequence of int
        Size of margin to drop from each side of the predicted patch in each dimension.
    stride : sequence of int
        Tile stride per spatial dimension. Must be positive and satisfy
        `stride[i] <= patch_size[i] - overlaps[i]`.
    """

    name: Literal["switi"] = "switi"
    """The name of the patching strategy."""

    stride: Annotated[Sequence[int], AfterValidator(all_positive(strict=True))] = Field(
        ...,
        min_length=2,
        max_length=3,
    )
    """Tile stride in each spatial dimension."""

    # TODO: can add halo here in future
    @field_validator("stride")
    @classmethod
    def dimensions_match(
        cls, value: Sequence[int] | None, info: ValidationInfo
    ):  # numpydoc ignore=RT01,PR01
        """Validate that dimensions match with those of the patch size."""
        if value is None:
            return value

        patch_size = cls._retrieve_patch_size(info)
        if len(value) != len(patch_size):
            raise ValueError(
                f"Must have the same number of dimensions as patch_size. "
                f"Got {len(value)} dimensions instead of {len(patch_size)} "
                "for patch_size."
            )
        return value

    @field_validator("stride")
    @classmethod
    def stride_compatible(
        cls, stride: Sequence[int], info: ValidationInfo
    ) -> Sequence[int]:  # numpydoc ignore=RT01,PR01
        """Validate `stride` against `patch_size` and `halo`.

        Each axis must satisfy `0 < stride[i] <= patch_size[i] - overlaps[i]`.
        If `overlaps` is None, treats it as all-zeros (degenerate sliding
        window without inner cropping).
        """
        patch_size = cls._retrieve_patch_size(info)
        overlaps = info.data.get("overlaps")

        effective_overlaps = overlaps if overlaps is not None else [0] * len(patch_size)
        for i, (p, o, s) in enumerate(
            zip(patch_size, effective_overlaps, stride, strict=True)
        ):
            if s > p - o:
                raise ValueError(
                    f"Axis {i}: stride ({s}) must be <= patch_size - overlaps "
                    f"({p} - {o} = {p - o})."
                )
        return stride

    @staticmethod
    def _retrieve_patch_size(info: ValidationInfo) -> Sequence[int]:
        """Retrieve the `patch_size` from `ValidationInfo`.

        For use within other validators, if the patch_size is not found an error is
        raised.

        Parameters
        ----------
        info : ValidationInfo
            Pydantic validation info object.

        Returns
        -------
        Sequence[int]
            The patch size.
        """
        patch_size = info.data.get("patch_size")
        if patch_size is None:
            raise ValueError(
                "Undefined patch_size. There may be validation errors in `patch_size`."
            )
        return patch_size

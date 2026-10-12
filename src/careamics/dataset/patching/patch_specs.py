"""A module to contain patch specification structures."""

from collections.abc import Sequence
from typing import TypedDict, TypeGuard, TypeVar

RegionSpecs = TypeVar("RegionSpecs", bound="PatchSpecs")


class PatchSpecs(TypedDict):
    """A dictionary that specifies a single patch in a series of `ImageStacks`.

    Attributes
    ----------
    data_idx: int
        Determines which `ImageStack` a patch belongs to, within a series of
        `ImageStack`s.
    sample_idx: int
        Determines which sample a patch belongs to, within an `ImageStack`.
    coords: sequence of int
        The top-left (and first z-slice for 3D data) of a patch. The sequence will have
        length 2 or 3, for 2D and 3D data respectively.
    patch_size: sequence of int
        The size of the patch. The sequence will have length 2 or 3, for 2D and 3D data
        respectively.
    """

    data_idx: int
    sample_idx: int
    coords: Sequence[int]
    patch_size: Sequence[int]


class TileSpecs(PatchSpecs):
    """A dictionary that specifies a single patch in a series of `ImageStacks`.

    Attributes
    ----------
    data_idx: int
        Determines which `ImageStack` a patch belongs to, within a series of
        `ImageStack`s.
    sample_idx: int
        Determines which sample a patch belongs to, within an `ImageStack`.
    coords: sequence of int
        The top-left (and first z-slice for 3D data) of a patch. The sequence will have
        length 2 or 3, for 2D and 3D data respectively.
    patch_size: sequence of int
        The size of the patch. The sequence will have length 2 or 3, for 2D and 3D data
        respectively.
    crop_coords: sequence of int
        The top-left side of where the tile will be cropped, in coordinates relative
        to the tile.
    crop_size: sequence of int
        The size of the cropped tile.
    stitch_coords: sequence of int
        Where the tile will be stitched back into an image, taking into account
        that the tile will be cropped, in coords relative to the image.
    total_tiles: int
        Number of tiles belonging to the same data.
    """

    crop_coords: Sequence[int]
    crop_size: Sequence[int]
    stitch_coords: Sequence[int]
    total_tiles: int


# not the most elegant solution but the patch specs are not used during training
class UncorrelatedPatchSpecs(PatchSpecs):
    """Each channel comes from a different location in the dataset.

    This kind of patch can be used in the MicroSplit training pipeline.

    Attributes
    ----------
    principal_channel : int
        An index that represents the channel that the main patch specs describe.
    all_data_idx : Sequence[int]
        A sequence that contains the data index that each channel patch belongs to.
    all_sample_idx : Sequence[int]
        A sequence that contains the sample index that each channel patch belongs to.
    all_coords : Sequence[Sequence[int]]
        The coordinate of each channel patch.
    """

    principal_channel: int
    all_data_idx: Sequence[int]
    all_sample_idx: Sequence[int]
    all_coords: Sequence[Sequence[int]]


def is_tile_specs(specs: PatchSpecs) -> TypeGuard[TileSpecs]:
    """Determine whether a given PatchSpecs is a TileSpecs.

    Used for type checking.

    Parameters
    ----------
    specs : PatchSpecs
        A patch specification.

    Returns
    -------
    bool
        Whether the given specs is a TileSpecs.
    """
    return (
        ("crop_coords" in specs)
        and ("crop_size" in specs)
        and ("stitch_coords" in specs)
    )


def is_uncorrelated_specs(specs: PatchSpecs) -> TypeGuard[UncorrelatedPatchSpecs]:
    """Determine whether a given PatchSpecs is an UncorrelatedPatchSpecs.

    Parameters
    ----------
    specs : PatchSpecs
        Patch specification to test.

    Returns
    -------
    bool
        Whether the patch specification contains uncorrelated patch metadata.
    """
    return (
        ("all_data_idx" in specs)
        and ("all_sample_idx" in specs)
        and ("all_coords" in specs)
    )

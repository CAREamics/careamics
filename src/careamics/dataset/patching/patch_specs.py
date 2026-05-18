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

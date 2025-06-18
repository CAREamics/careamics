"""A module to contain type definitions relating to patching strategies."""

from collections.abc import Sequence
from typing import Protocol, TypedDict


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
    """

    crop_coords: Sequence[int]
    crop_size: Sequence[int]
    stitch_coords: Sequence[int]


class PatchingStrategy(Protocol):
    """
    An interface for patching strategies.

    Patching strategies are a component of the `CAREamicsDataset`; they determine
    how patches are extracted from the underlying data.

    Attributes
    ----------
    n_patches: int
        The number of patches that the patching strategy will return.

    Methods
    -------
    get_patch_spec(index: int) -> PatchSpecs
        Get a patch specification for a given patch index.
    """

    @property
    def n_patches(self) -> int:
        """
        The number of patches that the patching strategy will return.

        It also determines the maximum index that can be given to `get_patch_spec`,
        and the length of the `CAREamicsDataset`.

        Returns
        -------
        int
            Number of patches.
        """
        ...

    def get_patch_spec(self, index: int) -> PatchSpecs:
        """
        Get a patch specification for a given patch index.

        This method is intended to be called from within the
        `CAREamicsDataset.__getitem__`. The index will be passed through from this
        method.

        Parameters
        ----------
        index : int
            A patch index.

        Returns
        -------
        PatchSpecs
            A dictionary that specifies a single patch in a series of `ImageStacks`.
        """
        ...

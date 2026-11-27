"""A module to contain patch specification structures."""

from collections.abc import Sequence
from itertools import islice
from typing import TypedDict, TypeGuard, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


class PerChannelPatchSpecs(PatchSpecs):
    """A patch whose channels each have their own location in the dataset.

    The channel locations can be the same or different. This kind of patch is used
    in the MicroSplit training pipeline.

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


def is_per_channel_specs(specs: PatchSpecs) -> TypeGuard[PerChannelPatchSpecs]:
    """Determine whether a given PatchSpecs is a PerChannelPatchSpecs.

    Parameters
    ----------
    specs : PatchSpecs
        Patch specification to test.

    Returns
    -------
    bool
        Whether the patch specification contains a location for each channel.
    """
    return (
        ("all_data_idx" in specs)
        and ("all_sample_idx" in specs)
        and ("all_coords" in specs)
    )


_PATCH_KIND = 0
_TILE_KIND = 1
_PER_CHANNEL_KIND = 2


def pack_patch_spec(spec: PatchSpecs) -> NDArray[np.int64]:
    """Pack a patch specification into a flat integer array.

    The array starts with the spec kind and the number of spatial dimensions, so
    `unpack_patch_spec` needs no other information. All specs of the same kind,
    number of spatial dimensions and number of channels have the same length.

    Parameters
    ----------
    spec : PatchSpecs
        A `PatchSpecs`, `TileSpecs` or `PerChannelPatchSpecs`.

    Returns
    -------
    NDArray[np.int64]
        The packed patch specification, 1D.
    """
    ndim = len(spec["patch_size"])
    values = [
        spec["data_idx"],
        spec["sample_idx"],
        *spec["coords"],
        *spec["patch_size"],
    ]
    if is_tile_specs(spec):
        kind = _TILE_KIND
        values += [
            *spec["crop_coords"],
            *spec["crop_size"],
            *spec["stitch_coords"],
            spec["total_tiles"],
        ]
    elif is_per_channel_specs(spec):
        kind = _PER_CHANNEL_KIND
        values += [
            len(spec["all_data_idx"]),
            spec["principal_channel"],
            *spec["all_data_idx"],
            *spec["all_sample_idx"],
            *(coord for coords in spec["all_coords"] for coord in coords),
        ]
    else:
        kind = _PATCH_KIND
    return np.array([kind, ndim, *values], dtype=np.int64)


def unpack_patch_spec(packed: ArrayLike) -> PatchSpecs:
    """Unpack a patch specification packed by `pack_patch_spec`.

    Parameters
    ----------
    packed : ArrayLike
        One packed patch specification, 1D. A tensor must be on the CPU.

    Returns
    -------
    PatchSpecs
        A `PatchSpecs`, `TileSpecs` or `PerChannelPatchSpecs`, with sequences as
        tuples.

    Raises
    ------
    ValueError
        If the packed spec kind is unknown.
    """
    values = iter(np.asarray(packed).tolist())
    kind, ndim = next(values), next(values)
    if kind not in (_PATCH_KIND, _TILE_KIND, _PER_CHANNEL_KIND):
        raise ValueError(f"Unknown packed patch spec kind: {kind}.")

    spec = PatchSpecs(
        data_idx=next(values),
        sample_idx=next(values),
        coords=tuple(islice(values, ndim)),
        patch_size=tuple(islice(values, ndim)),
    )
    if kind == _TILE_KIND:
        return TileSpecs(
            **spec,
            crop_coords=tuple(islice(values, ndim)),
            crop_size=tuple(islice(values, ndim)),
            stitch_coords=tuple(islice(values, ndim)),
            total_tiles=next(values),
        )
    if kind == _PER_CHANNEL_KIND:
        n_channels = next(values)
        return PerChannelPatchSpecs(
            **spec,
            principal_channel=next(values),
            all_data_idx=tuple(islice(values, n_channels)),
            all_sample_idx=tuple(islice(values, n_channels)),
            all_coords=tuple(tuple(islice(values, ndim)) for _ in range(n_channels)),
        )
    return spec

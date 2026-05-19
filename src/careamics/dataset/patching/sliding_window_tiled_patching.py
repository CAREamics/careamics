"""Sliding-window inner-tiled patching strategy."""

import itertools
from collections.abc import Sequence
from math import prod

from .patching import TileSpecs


def effective_mmse_count(patch_size: int, stride: int, overlap: int) -> int:
    """Per-axis effective MMSE count for `SlidingWindowTiledPatching`.

    Each interior pixel along the axis is covered by this many independent tile
    predictions (assuming `mmse_count = 1` in the model — each forward pass
    yields one stochastic draw). For a multi-axis pixel, the effective count is
    the product of this value across axes.

    Parameters
    ----------
    patch_size : int
        Tile size along the axis.
    stride : int
        Tile stride along the axis.
    overlap : int
        Overlap dropped from each adjacent tile pair (= 2 * margin per side).

    Returns
    -------
    int
        `max(1, (patch_size - overlap) // stride)`.
    """
    return max(1, (patch_size - overlap) // stride)


class SlidingWindowTiledPatching:
    """Patching strategy combining inner-tiling with sliding-window logic.

    Drops a margin `M = overlap // 2` from each side of each tile, with the
    same asymmetric edge convention as `TiledPatching` (first tile keeps its
    left side fully; last tile fills the remaining gap up to the image edge).
    Kept regions of adjacent interior tiles overlap when
    `stride < patch_size - overlap`, so each pure-interior pixel is covered by
    `K = (patch_size - overlap) // stride` independent predictions.

    When ``edge_replication=True`` (default), the first/last tile along each
    axis is replicated `K` times so border pixels also see `K` predictions,
    equalising the effective per-pixel MMSE count with the interior at the
    image edge. Corner tiles (edge along multiple axes simultaneously) are
    replicated multiplicatively (`K**d` for an `d`-axis corner) via the
    cartesian product.

    Note: there is a thin transition band of width `~(P - overlap)/2` just
    inside each image edge where interior coverage ramps down from `K` toward
    `0` but the replicated last tile hasn't kicked in yet. Pixels in that band
    see between `1` and `K-1` contributions. The interior and the absolute
    image edge are both at `K`; only the band in between is below.

    Intended to be used with posterior models configured with
    `mmse_count = 1`: each forward pass is one stochastic draw, so the
    per-pixel sample count is determined entirely by geometry.

    Parameters
    ----------
    data_shapes : sequence of (sequence of int)
        Shapes of the underlying data (axes SC(Z)YX).
    patch_size : sequence of int
        Tile size per spatial dimension (length 2 or 3).
    overlaps : sequence of int
        Overlap per spatial dimension (= 2 * margin). Must be even and strictly
        smaller than `patch_size[i]`.
    stride : sequence of int
        Tile stride per spatial dimension. Must satisfy
        `stride[i] <= patch_size[i] - overlaps[i]`.
    edge_replication : bool, default=True
        If True, replicate the first/last tile per axis `K` times so border
        pixels match interior MMSE coverage.
    """

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        overlaps: Sequence[int],
        stride: Sequence[int],
        edge_replication: bool = True,
    ):
        self.data_shapes = data_shapes
        self.patch_size = patch_size
        self.stride = stride
        self.overlaps = overlaps
        self.edge_replication = edge_replication
        self.tile_specs: list[TileSpecs] = self._generate_specs()

    @property
    def n_patches(self) -> int:
        """Total number of tile specs (post-replication)."""
        return len(self.tile_specs)

    def get_patch_spec(self, index: int) -> TileSpecs:
        """Return the tile specs for a given index.

        Parameters
        ----------
        index : int
            A patch index.

        Returns
        -------
        TileSpecs
            A dictionary that specifies a single patch in a series of `ImageStacks`.
        """
        return self.tile_specs[index]

    def get_patch_indices(self, data_idx: int) -> Sequence[int]:
        """
        Get the patch indices will return patches for a specific `image_stack`.

        The `image_stack` corresponds to the given `data_idx`.

        Parameters
        ----------
        data_idx : int
            An index that corresponds to a given `image_stack`.

        Returns
        -------
        sequence of int
            A sequence of patch indices, that when used to index the `CAREamicsDataset
            will return a patch that comes from the `image_stack` corresponding to the
            given `data_idx`.
        """
        return [
            i
            for i, spec in enumerate(self.tile_specs)
            if spec["data_idx"] == data_idx
        ]

    def _generate_specs(self) -> list[TileSpecs]:
        """Build the full list of tile spec.
        
        Returns
        -------
        list of TileSpecs
            Full list of tile specs.
        """
        tile_specs: list[TileSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]

            per_axis_expanded = [
                self._expand_1d(
                    spatial_shape[axis_idx],
                    self.patch_size[axis_idx],
                    self.stride[axis_idx],
                    self.overlaps[axis_idx],
                )
                for axis_idx in range(len(spatial_shape))
            ]

            n_tiles = prod(len(axis) for axis in per_axis_expanded) * data_shape[0]

            for sample_idx in range(data_shape[0]):
                for combo in itertools.product(*per_axis_expanded):
                    coords = tuple(c[0] for c in combo)
                    stitch_coords = tuple(c[1] for c in combo)
                    crop_coords = tuple(c[2] for c in combo)
                    crop_size = tuple(c[3] for c in combo)
                    tile_specs.append(
                        {
                            "data_idx": data_idx,
                            "sample_idx": sample_idx,
                            "coords": coords,
                            "patch_size": self.patch_size,
                            "crop_coords": crop_coords,
                            "crop_size": crop_size,
                            "stitch_coords": stitch_coords,
                            "total_tiles": n_tiles,
                        }
                    )

        return tile_specs

    def _expand_1d(
        self, axis_size: int, P: int, s: int, overlap: int
    ) -> list[tuple[int, int, int, int]]:
        """Build per-axis expanded tile list.

        Each entry is `(coord, stitch_coord, crop_coord, crop_size)`. Edge tiles
        are repeated `K = effective_mmse_count(P, s, overlap)` times when
        `edge_replication` is True; the outer cartesian product then yields
        multiplicative corner replication.
        """
        coords, stitch_coords, crop_coords, crop_size = self._compute_1d_coords(
            axis_size, patch_size=P, stride=s, overlap=overlap
        )
        n = len(coords)
        reps = [1] * n
        if self.edge_replication and n >= 2:
            K = effective_mmse_count(P, s, overlap)
            reps[0] = K
            reps[-1] = K

        expanded: list[tuple[int, int, int, int]] = []
        for k in range(n):
            entry = (coords[k], stitch_coords[k], crop_coords[k], crop_size[k])
            expanded.extend([entry] * reps[k])
        return expanded

    @staticmethod
    def _compute_1d_coords(
        axis_size: int, patch_size: int, stride: int, overlap: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Compute tile coordinates along a single axis.

        Mirrors the NG `TiledPatching._compute_1d_coords` exactly, with `stride`
        passed in as an independent parameter (rather than being locked to
        `patch_size - overlap`) and two `break` statements added so the
        boundary branches fire at most once when `stride < patch_size - overlap`.

        Parameters
        ----------
        axis_size : int
            The size of the axis.
        patch_size : int
            The tile size.
        stride : int
            The tile stride. Must satisfy `stride <= patch_size - overlap`.
        overlap : int
            The tile overlap.

        Returns
        -------
        coords: list of int
            The top-left (and first z-slice for 3D data) of a tile, in coords
            relative to the image.
        stitch_coords: list of int
            Where the tile will be stitched back into an image, taking into
            account that the tile will be cropped, in coords relative to the
            image.
        crop_coords: list of int
            The top-left side of where the tile will be cropped, in coordinates
            relative to the tile.
        crop_size: list of int
            The size of the cropped tile.
        """
        coords: list[int] = []
        stitch_coords: list[int] = []
        crop_coords: list[int] = []
        crop_size: list[int] = []

        for i in range(0, max(1, axis_size - overlap), stride):
            if i == 0:
                coords.append(i)
                crop_coords.append(0)
                stitch_coords.append(0)
                if axis_size <= patch_size:
                    crop_size.append(axis_size)
                    # Single-tile case: nothing more to emit. Without this
                    # break, the next iteration would fall into the else
                    # branch and emit a zero-size duplicate.
                    break
                else:
                    crop_size.append(patch_size - overlap // 2)
            elif (0 < i) and (i + patch_size < axis_size):
                coords.append(i)
                crop_coords.append(overlap // 2)
                stitch_coords.append(coords[-1] + crop_coords[-1])
                crop_size.append(patch_size - overlap)
            else:
                previous_crop_size = crop_size[-1] if crop_size else 1
                previous_stitch_coord = stitch_coords[-1] if stitch_coords else 0
                previous_tile_end = previous_stitch_coord + previous_crop_size

                coords.append(max(0, axis_size - patch_size))
                stitch_coords.append(previous_tile_end)
                crop_coords.append(stitch_coords[-1] - coords[-1])
                crop_size.append(axis_size - stitch_coords[-1])
                # Emit the last (gap-fill) tile exactly once. Subsequent
                # iterations would otherwise re-enter this branch and append
                # zero-size duplicates.
                break

        return (
            coords,
            stitch_coords,
            crop_coords,
            crop_size,
        )

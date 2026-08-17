"""Sliding-window inner-tiled patching strategy."""

import itertools
from collections.abc import Sequence
from math import prod

from .patch_specs import TileSpecs


class SwitiPatching:
    """Sliding-window inner-tiling patching strategy.

    Each pixel will be covered by `M` inner cropped tile regions, each at a an
    incrementally shifted tile location. These tiles are intended to be averaged
    together to create a smoother output than the MMSE estimate of inner tiling for
    posterior models.

    The number of border pixels (halo) to discard is controlled by the `overlaps`
    parameter; `overlaps // 2` pixels will be discarded from each side of the patch.

    The `stride` parameter controls by how many pixels each tile location is shifted by.

    Each pixel will be covered by the product of
    `(patch_size[i] - overlaps[i]) // stride[i]` over the dimensions `i`.

    See paper: https://arxiv.org/abs/2607.18990.

    Parameters
    ----------
    data_shapes : sequence of (sequence of int)
        Shapes of the underlying data (axes SC(Z)YX).
    patch_size : sequence of int
        Tile size per spatial dimension (length 2 or 3).
    overlaps : sequence of int
        A border of `overlaps[dim] // 2` is cropped from each side of the tile, in each
        dimension `dim`.
    stride : sequence of int
        Tile stride per spatial dimension. Must satisfy
        `stride[i] <= patch_size[i] - overlaps[i]`.
    """

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        overlaps: Sequence[int],
        stride: Sequence[int],
    ):
        """Sliding-window inner-tiling patching strategy.

        Each pixel will be covered by `M` inner cropped tile regions, each at a an
        incrementally shifted tile location. These tiles are intended to be averaged
        together to create a smoother output than the MMSE estimate of inner tiling for
        posterior models.

        The number of border pixels (halo) to discard is controlled by the `overlaps`
        parameter; `overlaps // 2` pixels will be discarded from each side of the patch.

        The `stride` parameter controls by how many pixels each tile location is shifted
        by.

        Each pixel will be covered by the product of
        `(patch_size[i] - overlaps[i]) // stride[i]` over the dimensions `i`.

        See paper: https://arxiv.org/abs/2607.18990.

        Parameters
        ----------
        data_shapes : sequence of (sequence of int)
            Shapes of the underlying data (axes SC(Z)YX).
        patch_size : sequence of int
            Tile size per spatial dimension (length 2 or 3).
        overlaps : sequence of int
            A border of `overlaps[dim] // 2` is cropped from each side of the tile, in
            each dimension `dim`.
        stride : sequence of int
            Tile stride per spatial dimension. Must satisfy
            `stride[i] <= patch_size[i] - overlaps[i]`.
        """
        self.data_shapes = data_shapes
        self.patch_size = patch_size
        self.stride = stride
        self.overlaps = overlaps
        self.tile_specs: list[TileSpecs] = self._generate_specs()

    @property
    def n_patches(self) -> int:
        """Total number of tile specs.

        Returns
        -------
        int
            Total number of patches.
        """
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
            i for i, spec in enumerate(self.tile_specs) if spec["data_idx"] == data_idx
        ]

    def _generate_specs(self) -> list[TileSpecs]:
        """Build the full list of tile specs.

        Returns
        -------
        list of TileSpecs
            Full list of tile specs.
        """
        tile_specs: list[TileSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]

            axis_specs: list[tuple[list[int], list[int], list[int], list[int]]] = [
                self._compute_1d_coords(
                    axis_size,
                    self.patch_size[axis_idx],
                    self.stride[axis_idx],
                    self.overlaps[axis_idx],
                )
                for axis_idx, axis_size in enumerate(spatial_shape)
            ]

            all_coords, all_stitch_coords, all_crop_coords, all_crop_size = zip(
                *axis_specs, strict=False
            )

            n_tiles = prod(len(dim) for dim in all_coords) * data_shape[0]

            for sample_idx in range(data_shape[0]):
                for coords, stitch_coords, crop_coords, crop_size in zip(
                    itertools.product(*all_coords),
                    itertools.product(*all_stitch_coords),
                    itertools.product(*all_crop_coords),
                    itertools.product(*all_crop_size),
                    strict=False,
                ):
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

    @staticmethod
    def _compute_1d_coords(
        axis_size: int, patch_size: int, stride: int, overlap: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Compute uniform-coverage sliding-window tile positions along one axis.

        Generates tile locations over the extent of the image by incremented coordinates
        by the `stride`. If a tile would theoretically partially lie outside the bounds
        of the image it is snapped to the nearest valid tile position.

        Yields uniform `K = (patch_size - overlap) // stride` coverage at every
        pixel.

        Parameters
        ----------
        axis_size : int
            The size of the axis.
        patch_size : int
            The tile size.
        stride : int
            The tile stride. Must satisfy `stride <= patch_size - overlap`.
        overlap : int
            The tile overlap (= 2 * margin).

        Returns
        -------
        coords : list of int
            Top-left position of the model input for each tile, in image coords.
        stitch_coords : list of int
            Where the cropped tile is stitched back into the image, in image
            coords.
        crop_coords : list of int
            Top-left of the kept region within the tile, in tile-local coords.
        crop_size : list of int
            Size of the kept region.
        """
        M = overlap // 2
        coords: list[int] = []
        stitch_coords: list[int] = []
        crop_coords: list[int] = []
        crop_size: list[int] = []

        if axis_size <= patch_size:
            return [0], [0], [0], [axis_size]

        # Smallest multiple of `stride` such that the conceptual kept region
        # [i+M, i+P-M) has non-empty intersection with [0, axis_size)
        lower_bound = M - patch_size + 1
        i_min = ((lower_bound + stride - 1) // stride) * stride

        i = i_min
        while True:
            kept_start = max(0, i + M)
            if kept_start >= axis_size:
                break
            kept_end = min(axis_size, i + patch_size - M)
            actual_coord = max(0, min(i, axis_size - patch_size))
            coords.append(actual_coord)
            stitch_coords.append(kept_start)
            crop_coords.append(kept_start - actual_coord)
            crop_size.append(kept_end - kept_start)
            i += stride

        return coords, stitch_coords, crop_coords, crop_size

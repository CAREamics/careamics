"Stratified patching strategy with the option to remove coordinates."

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Box = tuple[tuple[int, int], ...]


class _ImageStratifiedPatching:

    def __init__(
        self,
        shape: tuple[int, ...],
        patch_size: tuple[int, ...],
        rng: np.random.Generator | None = None,
    ):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.shape = shape
        self.patch_size = patch_size
        self.ndims = len(patch_size)

        # sampling regions will be stored in a dict
        # the keys correspond to a grid coordinate
        self.regions: dict[tuple[int, ...], _SamplingRegion] = {}
        self.areas: dict[tuple[int, ...], int] = {}

        self.grid_coords: NDArray[np.int_]
        self.grid_shape: tuple[int, ...]

        # define grid
        # we need shape + 1 to cover the last pixel when shape % patch_size = 0
        grid_axes: list[NDArray[np.int_]] = [
            np.arange(0, s + 1, ps) for s, ps in zip(shape, patch_size, strict=True)
        ]
        self.grid_coords = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1)
        self.grid_shape = self.grid_coords.shape[:-1]
        # populate the self.regions and self.areas dictionaries
        for grid_coord in self.grid_coords.reshape(-1, self.ndims):
            # find the pixel coordinate
            coord = grid_coord * np.array(self.patch_size)
            sampling_region = _SamplingRegion(
                tuple(coord.tolist()), self.patch_size, rng
            )
            sampling_region.clip(np.zeros(self.ndims, dtype=int), np.array(shape[2:]))
            # if the area is zero do not store the region
            if sum(sampling_region.areas) == 0:
                continue
            self.regions[tuple(grid_coord.tolist())] = sampling_region
            self.areas[tuple(grid_coord.tolist())] = sum(sampling_region.areas)


class _SamplingRegion:
    """
    Represent a small subregion that a patch can be sampled from.

    The region is double the patch size in each dimension. Quadrants or octants, for
    2D or 3D respectively, can be excluded from the region by calling the
    `remove_orthant` method.
    """

    def __init__(
        self,
        coord: tuple[int, ...],
        patch_size: tuple[int, ...],
        rng: np.random.Generator | None = None,
    ):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.coord = coord
        self.patch_size = patch_size
        self.ndims = len(patch_size)

        # A SamplingRegion is represented as sub regions
        # Having these regions makes it easier to remove orthants.
        # The 4 (2D) regions are represented by the diagram below
        # ┌───┬──────────────────────────┐
        # │   1                          │
        # ├─1─┼───────(patch_size)───────┤
        # │   │                          │
        #   ⋮              ⋮
        # └───┴──────────────────────────┘

        subregion_axis_extent = [((0, 1), (1, patch_size[i])) for i in range(2)]

        # a single subregion is represented by it's extent in each axis
        # An extent is a tuple (start, end), where the end is exclusive.
        self.subregions = list(itertools.product(*subregion_axis_extent))
        self.areas = self._calc_areas(self.subregions)

    def remove_orthant(
        self,
        orthant: tuple[Literal[0, 1], ...],
    ) -> None:
        orthant_region = tuple(
            (r, r + self.patch_size[i]) for i, r in enumerate(orthant)
        )
        self.subregions = [
            region
            for region in self.subregions
            if not _boxes_overlap(orthant_region, region)
        ]
        self.areas = self._calc_areas(self.subregions)

    def clip(
        self,
        minimum: np.ndarray[tuple[int], np.dtype[np.int_]],
        maximum: np.ndarray[tuple[int], np.dtype[np.int_]],
    ):
        minimum -= np.array(self.coord)
        maximum -= np.array(self.coord)
        subregions_clipped: list[tuple[tuple[int, int], ...]] = []

        # clip each sub region
        for region in self.subregions:
            start = np.array([axis_extent[0] for axis_extent in region])
            end = np.array([axis_extent[1] for axis_extent in region])

            start_clipped = np.clip(
                start, minimum, maximum - np.array(self.patch_size) + 1
            ).tolist()
            end_clipped = np.clip(
                end, minimum, maximum - np.array(self.patch_size) + 1
            ).tolist()
            subregions_clipped.append(
                tuple(zip(start_clipped, end_clipped, strict=True))
            )

        # remove any regions with zero area
        areas = self._calc_areas(subregions_clipped)
        subregions_clipped = [
            r for a, r in zip(areas, subregions_clipped, strict=True) if a > 0
        ]
        areas = [a for a in areas if a > 0]
        self.subregions = subregions_clipped
        self.areas = areas

    @staticmethod
    def _calc_areas(regions: Sequence[tuple[tuple[int, int], ...]]) -> list[int]:
        return [np.prod([r[1] - r[0] for r in region]).item() for region in regions]


def _boxes_overlap(
    box_a: Box,
    box_b: Box,
) -> bool:
    """
    Determine whether `box_a` and `box_b` overlap.
    """
    a_start = np.array([axis_extent[0] for axis_extent in box_a])
    a_end = np.array([axis_extent[1] for axis_extent in box_a])
    b_start = np.array([axis_extent[0] for axis_extent in box_b])
    b_end = np.array([axis_extent[1] for axis_extent in box_b])
    return (np.maximum(a_start, b_start) < np.minimum(a_end, b_end)).all().item()

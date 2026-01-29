"Stratified patching strategy with the option to exclude coordinates."

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .patching_strategy_protocol import PatchSpecs

Box = tuple[tuple[int, int], ...]


class StratifiedPatchingStrategy:

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed=seed)
        self.patch_size = patch_size
        self.data_shapes = data_shapes

        # create patching class for each sample in each image
        self.image_patching: list[list[_ImageStratifiedPatching]] = [
            [
                _ImageStratifiedPatching(shape[2:], patch_size, self.rng)
                for _ in range(shape[0])
            ]
            for shape in data_shapes
        ]

        # bins
        self.cumulative_image_patches, self.cumulative_sample_patches = (
            self._calc_bins()
        )

    @property
    def n_patches(self) -> int:
        return sum(
            sum([sample.n_patches for sample in image]) for image in self.image_patching
        )

    # TODO: add method to return valid grid coords for removal

    def exclude_patch(
        self, data_idx: int, sample_idx: int, grid_coord: tuple[int, ...]
    ):
        self.image_patching[data_idx][sample_idx].exclude_patch(grid_coord)

        # update bins after excluding patch
        self.cumulative_image_patches, self.cumulative_sample_patches = (
            self._calc_bins()
        )

    def get_patch_spec(self, index: int) -> PatchSpecs:

        # first find data_idx from cumulative image patches
        data_idx = np.digitize(index, self.cumulative_image_patches)
        data_bin_edge = (
            self.cumulative_image_patches[data_idx - 1] if data_idx != 0 else 0
        )

        image_sample_idx = np.digitize(data_bin_edge, self.cumulative_sample_patches)
        global_sample_idx = np.digitize(index, self.cumulative_sample_patches)
        sample_idx = global_sample_idx - image_sample_idx

        # now find index relative to start of sample for the stratified image sampler
        sample_bin_edge = (
            self.cumulative_sample_patches[global_sample_idx - 1]
            if global_sample_idx != 0
            else 0
        )
        index = index - sample_bin_edge

        coord = self.image_patching[data_idx][sample_idx].sample_patch_coord(index)
        return {
            "data_idx": data_idx.item(),
            "sample_idx": sample_idx.item(),
            "coords": tuple(coord.tolist()),
            "patch_size": self.patch_size,
        }

    # Note: this is used by the FileIterSampler
    def get_patch_indices(self, data_idx: int) -> Sequence[int]:
        patches_per_sample = [
            [sample.n_patches for sample in image] for image in self.image_patching
        ]
        patches_per_image = [sum(samples) for samples in patches_per_sample]
        cumulative_image_patches = np.cumsum(patches_per_image)

        start = 0 if data_idx == 0 else cumulative_image_patches[data_idx - 1]
        return np.arange(start, cumulative_image_patches[data_idx]).tolist()

    def _calc_bins(self) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        patches_per_sample = [
            [sample.n_patches for sample in image] for image in self.image_patching
        ]
        patches_per_image = [sum(samples) for samples in patches_per_sample]
        cumulative_image_patches = np.cumsum(patches_per_image)
        cumulative_sample_patches = np.cumsum(
            [n_patches for image in patches_per_sample for n_patches in image]
        )
        return cumulative_image_patches, cumulative_sample_patches


class _ImageStratifiedPatching:

    def __init__(
        self,
        shape: Sequence[int],
        patch_size: Sequence[int],
        rng: np.random.Generator | None = None,
    ):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.shape = shape
        self.patch_size = patch_size
        self.ndims = len(patch_size)
        # we need shape + 1 to cover the last pixel when shape % patch_size = 0
        self.grid_shape = tuple(
            ((np.array(shape) + 1) // np.array(patch_size)).tolist()
        )

        # sampling regions will be stored in a dict
        # the keys correspond to a grid coordinate
        self.regions: dict[tuple[int, ...], _SamplingRegion] = {}
        self.areas: dict[tuple[int, ...], int] = {}
        self.probs: dict[tuple[int, ...], float]
        self.bin_size: int
        self.bins: list[list[tuple[int, ...]]]
        self.n_patches: int

        # populate the self.regions and self.areas dictionaries
        for grid_coord in itertools.product(*[range(s) for s in self.grid_shape]):
            # find pixel coord
            coord = np.array(grid_coord) * np.array(patch_size)
            sampling_region = _SamplingRegion(tuple(coord), self.patch_size, rng)
            sampling_region.clip(np.zeros(self.ndims, dtype=int), np.array(shape))
            # if the area is zero do not store the region
            if sum(sampling_region.areas) == 0:
                continue
            self.regions[grid_coord] = sampling_region
            self.areas[grid_coord] = sum(sampling_region.areas)

        # no. of patches calculated from how many patches fit into the selectable area
        # this ensures that a pixel is expected to be selected 1 time per epoch
        # patches are packed into bins where no. of bins < no. of patches
        self._update_bins()

    def sample_patch_coord(self, index: int) -> NDArray[np.int_]:
        if index >= self.n_patches:
            raise ValueError(
                f"Index {index} out of bounds for image with {self.n_patches} patches."
            )

        # each index corresponds to a bin*,
        # regions within the selected bin are sampled proportionally to their area.
        # if there is remaining space in the bin it is assigned to all the regions
        # *contradicting the first statement, there may be fewer bins than patches
        # for empty bins, all regions are sampled with a prob proportional to area
        if index < len(self.bins):
            bin_ = self.bins[index]
        else:
            bin_ = []

        region = self._sample_region_from_bin(bin_)
        return region.sample_patch_coord()

    def _sample_region_from_bin(self, bin: list[tuple[int, ...]]) -> "_SamplingRegion":
        keys = list(self.regions.keys())
        probs = np.array([self.probs[key] for key in keys])
        if len(bin) != 0:
            indices = np.where(
                (np.array(keys) == np.array(bin)[:, None]).all(2).any(0)
            )[0]
        else:
            indices = np.array([], dtype=int)
        weights = np.zeros_like(probs)
        weights[indices] = probs[indices]
        remaining_prob = 1 - weights.sum()
        weights += probs / probs.sum() * remaining_prob
        selected_key_idx = self.rng.choice(np.arange(len(keys)), p=weights)
        selected_key = keys[selected_key_idx]
        return self.regions[selected_key]

    def exclude_patch(self, grid_coord: tuple[int, ...]):
        d: tuple[Literal[0, 1], ...] = (0, 1)
        for d_idx in itertools.product(*[d for _ in range(self.ndims)]):
            q: tuple[Literal[0, 1], ...] = tuple(0 if i == 1 else 1 for i in d_idx)
            grid_idx = tuple(
                g - (1 - i) for g, i in zip(grid_coord, d_idx, strict=True)
            )
            if grid_idx not in self.regions:
                continue
            self.regions[grid_idx].exclude_orthant(q)
            self.areas[grid_idx] = sum(self.regions[grid_idx].areas)

            if self.areas[grid_idx] == 0:
                del self.regions[grid_idx]
                del self.areas[grid_idx]
                del self.probs[grid_idx]

        self._update_bins()

    def _update_bins(self):
        self.n_patches = int(
            np.ceil(sum(self.areas.values()) / np.prod(self.patch_size))
        )
        self.bin_size, _ = _find_bin_size(self.areas, self.n_patches)
        self.bins = _region_bin_packing(self.areas, self.bin_size)
        self.probs = {key: area / self.bin_size for key, area in self.areas.items()}


class _SamplingRegion:
    """
    Represent a small subregion that a patch can be sampled from.

    The region is double the patch size in each dimension. Quadrants or octants, for
    2D or 3D respectively, can be excluded from the region by calling the
    `exclude_orthant` method.
    """

    def __init__(
        self,
        coord: Sequence[int],
        patch_size: Sequence[int],
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

        subregion_axis_extent = [
            ((0, 1), (1, patch_size[i])) for i in range(self.ndims)
        ]

        # a single subregion is represented by it's extent in each axis
        # An extent is a tuple (start, end), where the end is exclusive.
        self.subregions = list(itertools.product(*subregion_axis_extent))
        self.areas = self._calc_areas(self.subregions)

    def sample_patch_coord(self) -> NDArray[np.int_]:
        areas = np.array(self.areas)
        # first a region is chosen (proportionally to area)
        r_idx = self.rng.choice(np.arange(len(self.areas)), p=areas / areas.sum())
        region = self.subregions[r_idx]
        # then a coordinate is chosen
        start = np.array([r[0] for r in region])
        end = np.array([r[1] for r in region])

        return self.rng.integers(start, end) + np.array(self.coord)

    def exclude_orthant(
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


# --- helper funcs


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


def _region_bin_packing(
    areas: dict[tuple[int, ...], int], bin_size: int
) -> list[list[tuple[int, ...]]]:
    if len(areas) == 0:
        return []

    sorted_keys = sorted(areas.keys(), key=lambda k: areas[k], reverse=True)
    bins: list[list[tuple[int, ...]]] = []

    for key in sorted_keys:
        # Find the bin with least remaining space that still fits this value
        best_bin_idx = None
        min_remaining_space = bin_size + 1

        for i, bin_contents in enumerate(bins):
            current_sum = sum([areas[k] for k in bin_contents])
            remaining_space = bin_size - current_sum

            # Can it fit? Is it tighter than our current best?
            value = areas[key]
            if remaining_space >= value and remaining_space < min_remaining_space:
                best_bin_idx = i
                min_remaining_space = remaining_space

        # Add to best bin or create new one
        if best_bin_idx is not None:
            bins[best_bin_idx].append(key)
        else:
            bins.append([key])

    return bins


def _find_bin_size(
    areas: dict[tuple[int, ...], int], target_n_bins: int
) -> tuple[int, int]:
    if len(areas) == 0:
        return 0, 0

    # Binary search bounds
    min_size = max(areas.values())
    max_size = sum(areas.values())

    # Edge case: if we want 1 bin, we need bin size = sum of all values
    if target_n_bins == 1:
        return max_size, 1

    # Edge case: if we want as many bins as values, bin size = max value
    if target_n_bins >= len(areas):
        return min_size, len(areas)

    best_bin_size = None
    best_num_bins = None

    while min_size <= max_size:
        mid_size = (min_size + max_size) // 2

        # Test how many bins we get with this size
        bins = _region_bin_packing(areas, mid_size)
        num_bins = len(bins)

        if num_bins == target_n_bins:
            # Found exact match! Save it and keep searching for smaller bin size
            best_bin_size = mid_size
            best_num_bins = num_bins
            max_size = mid_size - 1  # Try smaller bin size
        elif num_bins > target_n_bins:
            # Too many bins, need larger bin size
            min_size = mid_size + 1
        else:
            # Too few bins (num_bins < target_bins)
            # Save as potential answer if it's our best so far
            if best_num_bins is None or num_bins > best_num_bins:
                best_bin_size = mid_size
                best_num_bins = num_bins
            elif (
                best_bin_size is not None
                and num_bins == best_num_bins
                and mid_size < best_bin_size
            ):
                # Same number of bins but smaller size - better!
                best_bin_size = mid_size

            # Try smaller bin size to increase bins
            max_size = mid_size - 1

    best_bin_size = 0 if best_bin_size is None else best_bin_size
    best_num_bins = 0 if best_num_bins is None else best_num_bins

    return best_bin_size, best_num_bins

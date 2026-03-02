"Stratified patching strategy with the option to exclude coordinates."

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .patching_strategy_protocol import PatchSpecs

Box = tuple[tuple[int, int], ...]

# --- Structure overview
# Sampling regions which have an area of double the patch size are created so that they
# lie on a grid with a spacing equal to the patch size. This ensures that all possible
# patch coordinates can be selected. These sampling regions are represented by the
# `_SamplingRegion` class. The sampling region itself contains subregions, which make
# it easier to exclude patch regions.

# For each sample in each image-stack an `_ImageStratifiedPatching` class is created,
# this stores `_SamplingRegions` in a `dict[tuple[int, ...], _SamplingRegion]` where the
# key of the dictionary represents the grid coordinate that the top-left corner of the
# sampling region lies on. Most of the patching logic happens in this class.

# The `StratifiedPatchingStrategy` stores a `list[list[_ImageStratifiedPatching]]` where
#  the elements of the outer list represent each image stack and there is an
# `_ImageStratifiedPatching` instance for each sample in each image stack.


class StratifiedPatchingStrategy:
    """
    A stratified patching strategy that also allows patches on a grid to be excluded.

    Patches will be sampled from sampling regions that are two times the patch size in
    each dimension. Some sampling regions may be smaller than this because they are on
    the edge of an image or because a nearby patch has been excluded.

    If the same index is used twice to sample a patch with the method `get_patch_spec`
    there will be a high probability that it will come from the same sampling region,
    but not necessarily 100%. Smaller sampling regions may be binned together into a
    single index. The mean of all the expected values that each pixel will be selected
    in a patch per epoch is 1.

    The number of patches is determined from the number of selectable patch coordinates.
    """

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        seed: int | None = None,
    ):
        """
        A patching strategy for sampling stratified patches.

        Parameters
        ----------
        data_shapes : sequence of (sequence of int)
            The shapes of the underlying data. Each element is the dimension of the
            axes SC(Z)YX.
        patch_size : sequence of int
            The size of the patch. The sequence will have length 2 or 3, for 2D and 3D
            data respectively.
        seed : int, optional
            An optional seed to ensure the reproducibility of the random patches.
        """
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
        (
            self.cumulative_image_patches,
            self.cumulative_sample_patches,
            self.cumulative_image_samples,
        ) = self._calc_bins()

    @property
    def n_patches(self) -> int:
        """
        The number of patches that this patching strategy will return.

        It also determines the maximum index that can be given to `get_patch_spec`.
        """
        return sum(
            sum([sample.n_patches for sample in image]) for image in self.image_patching
        )

    def exclude_patches(
        self, data_idx: int, sample_idx: int, grid_coords: Sequence[tuple[int, ...]]
    ):
        """
        Exclude patches from being sampled.

        Excluded patches must lie on a grid which starts at (0, 0) and has a spacing of
        the given `patch_size`.

        After calling this method the number of patches will be recalculated and the
        excluded patches will never be returned by `get_patch_spec`.

        Parameters
        ----------
        data_idx : int
            The index of the "image stack" that the patches will be excluded from.
        sample_idx : int
            An index that corresponds to the sample in the "image stack" that the
            patches will be excluded from.
        grid_coords : Sequence[tuple[int, ...]]
            A sequence of 2D or 3D tuples. Each tuple corresponds to a grid coordinate
            that will be excluded from sampling. The grid starts at (0, 0) and has a
            spacing of the given `patch_size`.
        """
        self.image_patching[data_idx][sample_idx].exclude_patches(grid_coords)

        # update bins after excluding patch
        (
            self.cumulative_image_patches,
            self.cumulative_sample_patches,
            self.cumulative_image_samples,
        ) = self._calc_bins()

    def get_patch_spec(self, index: int) -> PatchSpecs:
        """Return the patch specs for a given index.

        Parameters
        ----------
        index : int
            A patch index.

        Returns
        -------
        PatchSpecs
            A dictionary that specifies a single patch in a series of `ImageStacks`.
        """

        # first find data_idx from cumulative image patches
        data_idx = np.digitize(index, self.cumulative_image_patches)

        # find the sample idx from the cumulative-sample-patches
        # and the cumulative-image-samples
        global_sample_idx = np.digitize(index, self.cumulative_sample_patches)
        image_sample_idx = (
            self.cumulative_image_samples[data_idx - 1] if data_idx != 0 else 0
        )
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
        """
        Return the patch indices for a specific `image_stack`.

        The `image_stack` corresponds to the given `data_idx`.

        Parameters
        ----------
        data_idx : int
            An index that corresponds to a given `image_stack`.

        Returns
        -------
        sequence of int
            A sequence of patch indices, used to index the `CAREamicsDataset`
            to return a patch that comes from the `image_stack` corresponding to the
            given `data_idx`.
        """
        start = 0 if data_idx == 0 else self.cumulative_image_patches[data_idx - 1]
        return np.arange(start, self.cumulative_image_patches[data_idx]).tolist()

    def get_included_grid_coords(self) -> dict[tuple[int, int], list[tuple[int, ...]]]:
        """
        Get all grid coordinates included in the patching strategy.

        If a grid coordinate is not included, a patch can never be selected from the
        region `[grid_coord*patch_size, (grid_coord+1)*patch_size]`.

        Returns
        -------
        grid_coords : dict[tuple[int, int], list[tuple, ...]]
            The key of the returned dictionary corresponds to the
            `(data_idx, sample_idx)` and the values are the corresponding grid coords.
        """
        included_grid_coords: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        for data_idx, image_patch_list in enumerate(self.image_patching):
            for sample_idx, sample_patching in enumerate(image_patch_list):
                included_grid_coords[(data_idx, sample_idx)] = (
                    sample_patching.get_included_grid_coords()
                )
        return included_grid_coords

    def _calc_bins(self) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
        """
        Calculate bins to determine which image and sample a patch index maps to.

        Returns
        ------
        cumulative_image_patches : numpy.ndarray
            Bins that show which "image stack" a patch index belongs to.
        cumulative_sample_patches : numpy.ndarray
            Bins that show which sample a patch index belongs to.
        cumulative_image_samples : numpy.ndarray
            Bins that show which "image stack" a sample belongs to.
        """
        patches_per_sample = [
            [sample.n_patches for sample in image] for image in self.image_patching
        ]
        patches_per_image = [sum(samples) for samples in patches_per_sample]
        cumulative_image_patches = np.cumsum(patches_per_image)
        cumulative_sample_patches = np.cumsum(
            [n_patches for image in patches_per_sample for n_patches in image]
        )

        samples_per_image = [len(samples) for samples in patches_per_sample]
        cumulative_image_samples = np.cumsum(samples_per_image)

        return (
            cumulative_image_patches,
            cumulative_sample_patches,
            cumulative_image_samples,
        )


class _ImageStratifiedPatching:
    """A class used to sample the patch coordinates for a single sample.

    The number of patches is determined from the number of selectable patch coordinates.

    Sampling regions have a size of 2 times the patch size in each dimension, unless
    the region is near an edge or a nearby patch that has been excluded.

    Sampling regions are packed into bins to achieve the desired number of patches.
    Each index now corresponds to a bin, the probability that a region in the bin is
    sampled is equal to the ratio of the area of the region to the bin size. If there
    is space left in the bin this remaining probability is used to give a small chance
    that any of the regions in the image may be sampled.

    Attributes
    ----------
    regions : dict[tuple[int, ...], _SamplingRegion]
        The sampling regions that patch coordinates are sampled from. The sampling
        regions lie on a grid, and the key of the dictionary corresponds to the
        grid coordinate of the sampling region.
    areas : dict[tuple[int, ...], int]
        The selectable patch coordinates in each sampling region.
    probs : dict[tuple[int, ...], float]
        The probability of selecting a sub region from its bin, this is just its area
        divided by the bin size.
    excluded_patches : list[tuple[int, ...]]
        A list of grid coordinates that correspond to the patches that have been
        excluded.
    bin_size : int
        The size of the bins that the sampling regions are packed into.
    bins : list[list[tuple[int, ...]]]
        A single bin contains sampling regions, it is represented as a list of
        grid coordinates that correspond to a sampling region. All the bins are stored
        in a list.
    n_patches : int
        The number of patches.
    """

    def __init__(
        self,
        shape: Sequence[int],
        patch_size: Sequence[int],
        rng: np.random.Generator | None = None,
    ):
        """
        A class used to sample the patch coordinates for a single sample.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the image with axes (Z)YX.
        patch_size : Sequence[int]
            The chosen patch size with axes (Z)YX.
        rng : numpy.random.Generator, optional
            A numpy random number generator.
        """
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

        self.excluded_patches: set[tuple[int, ...]] = set()
        self.bin_size: int
        self.bins: list[list[tuple[int, ...]]]
        self.n_patches: int

        # populate the self.regions and self.areas dictionaries
        for grid_coord in itertools.product(*[range(s) for s in self.grid_shape]):
            # find pixel coord
            coord = np.array(grid_coord) * np.array(patch_size)
            sampling_region = _SamplingRegion(tuple(coord), self.patch_size, rng)
            # sampling regions are clipped to not overlap the bounds of the image
            # this is necessary when the image size is not divisible by the patch size
            sampling_region.clip(np.zeros(self.ndims, dtype=int), np.array(shape))
            # if the area is zero do not store the region
            if sum(sampling_region.areas) == 0:
                continue
            self.regions[grid_coord] = sampling_region
            self.areas[grid_coord] = sum(sampling_region.areas)

        # no. of patches calculated from how many patches fit into the selectable area
        # this ensures that a pixel is expected to be selected 1 time per epoch
        # patches are packed into bins where no. of bins < no. of patches
        self.n_patches, self.bin_size, self.bins, self.probs = (
            self._recalculate_sampling()
        )

    def sample_patch_coord(self, index: int) -> NDArray[np.int_]:
        """
        Sample a patch coordinate for a given index.

        Parameters
        ----------
        index : int
            Corresponds with high probability to a patch from a particular region, or
            multiple regions.
        """
        if index >= self.n_patches:
            raise ValueError(
                f"Index {index} out of bounds for image with {self.n_patches} patches."
            )

        # the number of bins will be less than or equal to the number of patches,
        #
        # for index < no. of bins, the index will select a bin.
        # A sampling region will be selected from the bin with a calculated probability,
        # the probability is proportional to the no. of region's selectable coordinates.
        # A bin might not be perfectly filled, the remaining probability has to be used,
        # it is assigned to all the sampling regions proportionally to their area.
        #
        # for index > no. of bins, it is effectively treated as an empty bin
        # all of sampling regions are selected proportionally to their area.

        if index < len(self.bins):
            bin_ = self.bins[index]
        else:
            bin_ = []

        region = self._sample_region_from_bin(bin_)
        return region.sample_patch_coord()

    def _sample_region_from_bin(self, bin: list[tuple[int, ...]]) -> "_SamplingRegion":
        """
        Sample a region from a given bin. Bins can contain multiple sampling regions.

        bin : list[tuple[int, ...]]
            A bin of sampling regions represented by a list of grid coordinates. Each
            grid coordinate corresponds to one sampling region.
        """
        grid_coords = list(self.regions.keys())
        probs = np.array([self.probs[key] for key in grid_coords])
        if len(bin) != 0:
            indices = np.where(
                # behaves like isin but for multiple values
                # i.e. finding the indices where grid_coords == bin
                (np.array(grid_coords) == np.array(bin)[:, None])
                .all(2)
                .any(0)
            )[0]
        else:
            indices = np.array([], dtype=int)
        # The ratio of the area of a region to the size of the bin is equal to the
        # probability that the region will be sampled.
        weights = np.zeros_like(probs)
        weights[indices] = probs[indices]

        # remaining space in the bin must be used up
        # this is assigned to all the regions in the image
        remaining_prob = 1 - weights.sum()
        weights += probs / probs.sum() * remaining_prob

        # the region is sampled using these calculated weights.
        selected_key_idx = self.rng.choice(np.arange(len(grid_coords)), p=weights)
        selected_key = grid_coords[selected_key_idx]
        return self.regions[selected_key]

    def exclude_patches(self, grid_coords: Sequence[tuple[int, ...]]):
        """
        Exclude patches from being sampled.

        Excluded patches must lie on a grid which starts at (0, 0) and has a spacing of
        the given `patch_size`.

        After calling this method the number of patches will be recalculated and the
        excluded patches will never be returned by `sample_patch_coord`.

        Parameters
        ----------
        grid_coords : Sequence[tuple[int, ...]]
            A sequence of 2D or 3D tuples. Each tuple corresponds to a grid coordinate
            that will be excluded from sampling. The grid starts at (0, 0) and has a
            spacing of the given `patch_size`.
        """
        self.excluded_patches.update(grid_coords)
        for grid_coord in grid_coords:
            d: tuple[Literal[0, 1], ...] = (0, 1)
            # exclude the patch from all the sampling regions that cover it
            # These are the grid coords at: subtract (0, 0), (0, 1), (1, 0) (for 2D)
            for d_idx in itertools.product(*[d for _ in range(self.ndims)]):
                # q is the ID of the orthant to remove
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

        self.n_patches, self.bin_size, self.bins, self.probs = (
            self._recalculate_sampling()
        )

    def get_included_grid_coords(self) -> list[tuple[int, ...]]:
        """
        Get all the included grid coordinates in the patching strategy.

        If a grid coordinate is not included, a patch can never be selected from the
        region `[grid_coord*patch_size, (grid_coord+1)*patch_size]`.

        Returns
        -------
        grid_coords : list[tuple, ...]]
            The list of included grid coordinates.
        """
        grid_coords_all: set[tuple[int, ...]] = set(self.regions.keys())
        return list(grid_coords_all.difference(self.excluded_patches))

    def _recalculate_sampling(self):
        """
        Recalculate how patches will be sampled.

        Returns
        -------
        n_patches : int
            The number of patches.
        bin_size : int
            The size of the bins that sampling regions are packed into. Each bin
            corresponds to a sampling index.
        bins : list[list[tuple[int, ...]]]
            Bins containing sampling regions. The regions are packed based on their
            area.
        probs : dict[tuple[int, ...], float]
            The probability that a sampling region will be selected from its bin.
        """

        # NOTE: alternative number of patches:
        # - results in expected value of pixel not near edge being 1 per epoch.
        # n_patches = int(np.ceil(sum(self.areas.values()) / np.prod(self.patch_size)))

        # have to guard for case that area is zero. It is possible to not have any
        # selectable patches when only edge regions are remaining.
        if sum(self.areas.values()) != 0:
            # NOTE: taking prod first vs taking ceil first
            # taking ceil first is more similar to random patching
            # for now have chosen to take prod first

            # total_patches = int(
            #     np.prod(np.ceil(np.array(self.shape) / np.array(self.patch_size)))
            # )
            total_patches = int(
                np.ceil(np.prod(np.array(self.shape) / np.array(self.patch_size)))
            )
            n_patches = total_patches - len(self.excluded_patches)
        else:
            n_patches = 0

        bin_size, _ = _find_bin_size(self.areas, n_patches)
        bins = _region_bin_packing(self.areas, bin_size)
        probs = {key: area / bin_size for key, area in self.areas.items()}
        return n_patches, bin_size, bins, probs


class _SamplingRegion:
    """
    Represent a small subregion that a patch can be sampled from.

    The region is double the patch size in each dimension. Quadrants or octants, for
    2D or 3D respectively, can be excluded from the region by calling the
    `exclude_orthant` method.

    Attributes
    ----------
    coord : Sequence[int]
        The top-left (and depth) coordinate of the sampling region, length 2 or 3 for
        2D or 3D respectively.
    patch_size : Sequence[int]
        The patch size.
    ndims : int
        The number of dimension, 2 or 3.
    subregions : list[tuple[tuple[int, int], ...]]
        A list of subregions that are represented by an extent in each dimensions. For
        example ((0, 16), (0, 32)) would be the 2D box with the extent 0 to 16 in the
        first dimension and 0 to 32 in the 2nd dimension. Note, this is the region of
        selectable patch coordinates.
    areas : list[int]
        The number of selectable patch coordinates in each sub-region.
    """

    def __init__(
        self,
        coord: Sequence[int],
        patch_size: Sequence[int],
        rng: np.random.Generator | None = None,
    ):
        """
        Represent a small subregion that a patch can be sampled from.

        Parameters
        ----------
        coord : Sequence[int]
            The top-left (and depth) coordinate of the sampling region, length 2 or 3
            for 2D or 3D respectively.
        patch_size : Sequence[int]
            The patch size, a sampling region will have double the patch size in each
            dimension.
        rng: numpy.random.Generator, optional
            A numpy random number generator.
        """
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
        # ├─1─┼─────(patch_size - 1)─────┤
        # │   │                          │
        #   ⋮              ⋮
        # └───┴──────────────────────────┘

        subregion_axis_extent = [
            ((0, 1), (1, patch_size[i])) for i in range(self.ndims)
        ]

        # a single subregion is represented by its extent in each axis
        # An extent is a tuple (start, end), where the end is exclusive.
        self.subregions = list(itertools.product(*subregion_axis_extent))
        self.areas = self._calc_areas(self.subregions)

    def sample_patch_coord(self) -> NDArray[np.int_]:
        """Sample a patch coordinate from the sampling region."""
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
        """
        Exclude an orthant from the sampling region.

        (An orthant is a quadrant or octant for 2D and 3D respectively).

        Parameters
        ----------
        orthant: tuple[{0, 1}, ...]
            A 2D or 3D tuple of 0s and 1s which identify orthants, e.g. (0, 0) would be
            the top left quadrant and (0, 1) would be the top right quadrant.
        """
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
        """
        Clip a sampling region.

        After clipping a selected patch will not overlap with the `minimum` or `maximum`
        coordinate.

        Parameters
        ----------
        minimum : numpy.ndarray[np.int_]
            An array of length 2 or 3 that represents the minimum coordinate at
            which a patch can be defined.
        maximum : numpy.ndarray[np.int_]
            An array of length 2 or 3 that represents the maximum coordinate at
            which a patch can be defined.
        """
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


# bin packing using the best-fit-decreasing algorithm
def _region_bin_packing(
    areas: dict[tuple[int, ...], int], bin_size: int
) -> list[list[tuple[int, ...]]]:
    """Pack regions in bins with `bin_size` based on their area."""
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


# performs a binary search
def _find_bin_size(
    areas: dict[tuple[int, ...], int], target_n_bins: int
) -> tuple[int, int]:
    """Find the minimum bin size that will result in `target_n_bins` or less.

    Parameters
    ----------
    areas : dict[tuple[int, ...], int]
        A dictionary of sampling region areas. The key is the grid coordinate of the
        corresponding sampling region. These are the sampling regions created by
        `_ImageStratifiedPatching`.
    target_n_bins : int
        The desired number of bins.

    Returns
    -------
    best_bin_size : int
        The bin size found to result in `target_n_bins` number of bins or less.
    best_num_bins : int
        The resulting number of bins.
    """
    if len(areas) == 0:
        return 0, 0

    # Binary search bounds
    min_size = max(areas.values())
    max_size = sum(areas.values())

    # Edge case: if we want 1 bin, we need bin size = sum of all region areas
    if target_n_bins == 1:
        return max_size, 1

    # Edge case: if we want as many bins as there are regions, bin size = maximum area
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

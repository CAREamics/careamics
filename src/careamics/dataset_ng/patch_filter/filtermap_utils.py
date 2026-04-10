"""Filter map utilities."""

import itertools
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from careamics.dataset_ng.factory import init_patch_extractor
from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patching_strategies import StratifiedPatchingStrategy

FilterValueFunc = Callable[[NDArray[Any]], float]
"""A function that outputs a value to determine whether a patch should be filtered."""


def create_filter_map(
    image: np.ndarray,
    filter_value_func: FilterValueFunc,
    patch_size: tuple[int, ...],
    direction: Literal["greater", "less"] = "greater",
):
    """
    Create a filter map to assess the correct thresholds for filtering.

    Parameters
    ----------
    image : numpy.ndarray
        The image to create a filter map for, it can be 2D or 3D with no channels. E.g.
        it should have the axes `YX` or `ZYX`.
    filter_value_func : FilterValue
        A function that produces a single value for a patch to create the filter map
        with.
    patch_size : tuple of int
        The patch size intended for training.
    direction : {"greater", "less"}
        Whether the filter will take thresholds greater than or less than the filter
        value. "greater" indicates values above the threshold will be kept for training.

    Returns
    -------
    np.ndarray
        An array that has the same shape as the image. It shows regions of the image
        that will be filtered above or below some threshold.
    """
    n_dims = len(patch_size)
    region_size = tuple(ps * 2 for ps in patch_size)
    patching_strategy = StratifiedPatchingStrategy(
        [(1, 1, *image.shape)], patch_size=patch_size
    )
    input_extractor = init_patch_extractor(
        PatchExtractor, load_arrays, [image], axes="YX"
    )

    # the stratified sampling regions regions overlap
    # 4 overlaping regions for 2D 8 overlapping regions for 3D
    filtermaps = np.zeros((2**n_dims, *image.shape))
    all_grid_coords = patching_strategy.get_all_grid_coords().items()

    for (data_idx, sample_idx), grid_coords in all_grid_coords:
        for coords in grid_coords:
            region_coords = tuple(
                ps * c for ps, c in zip(patch_size, coords, strict=True)
            )
            patch = input_extractor.extract_channel_patch(
                data_idx,
                sample_idx=sample_idx,
                channels=[0],
                coords=region_coords,
                patch_size=region_size,
            )
            value = filter_value_func(patch)

            for i, orth_coords in enumerate(
                list(itertools.product(*[[0, 1]] * n_dims))
            ):
                filtermaps[
                    i,
                    *[
                        slice(c + x * ps, c + (x + 1) * ps)
                        for x, c, ps in zip(
                            orth_coords, region_coords, patch_size, strict=True
                        )
                    ],  # type: ignore
                ] = value

    if direction == "greater":
        filtermap = filtermaps.max(axis=0)
    else:
        filtermap = filtermaps.min(axis=0)

    return filtermap

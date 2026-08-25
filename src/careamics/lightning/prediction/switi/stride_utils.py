# TODO: move to config?
"""Utility functions to calculate the stride for Sliding Window Inner Tiling (SWITi).

The stride is optimized so that the number of tiles covering each pixel best matches the
desired MMSE count.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brute

from careamics.utils import get_logger

logger = get_logger(__name__)


def effective_mmse_count(patch_size: int, stride: int, overlap: int) -> int:
    """Calculate per-axis effective MMSE count for `SwitiPatching`.

    The effective MMSE count for multiple axes is the product of the result per axis.

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
        The number of tiles each pixel would be covered by along an axis given the
        parameters.
    """
    return int(np.ceil((patch_size - overlap) / stride))


def compute_switi_stride(
    patch_size: Sequence[int],
    overlap: Sequence[int],
    target_mmse_count: int,
    *,
    stride_z: int | None = None,
) -> tuple[tuple[int, int] | tuple[int, int, int], int]:
    """
    Compute the optimum stride in each spatial dimension to achieve a target MMSE count.

    The stride in X and Y will be equal.

    If the target MMSE count is not exactly achievable, the achieved MMSE count will
    always be greater than the target.

    Parameters
    ----------
    patch_size : Sequence[int]
        The input tile size, either 2D or 3D.
    overlap : Sequence[int]
        Describes the border cropped from each side of the tile, the border is
        `overlaps[dim] // 2` for each dimension `dim`.
    target_mmse_count : int
        Target MMSE count, e.g. how many tiles will cover each pixel.
    stride_z : int | None, default=None
        Optionally choose to manually fix the z_stride, ignored for 2D, if None for 3D
        it will be calculated. The effective MMSE count will be optimized by jointly
        tuning `stride_z` and the stride in xy.

    Returns
    -------
    tuple[int, int] | tuple[int, int, int]
        The calculated stride, 2D or 3D depending on the input.
    int
        The effective MMSE resulting from the calculated stride.

    Raises
    ------
    ValueError
        If the patch size is not square in XY.
    ValueError
        If the overlaps are not equal in XY.
    ValueError
        If the `patch_size` is not valid (not 2D or 3D).
    """
    # NOTE: should already be validated in config
    if patch_size[-1] != patch_size[-2]:
        raise ValueError(
            f"Only patches square in the XY dimensions are valid, got {patch_size}."
        )
    if overlap[-2] != overlap[-1]:  # TODO: not sure if validated in config
        raise ValueError(f"Overlaps must be equal in XY, got {overlap}.")

    crop_size_xy = patch_size[-2] - overlap[-1]  # equal for X and Y
    stride: tuple[int, int] | tuple[int, int, int]
    match patch_size, stride_z:
        case (_, _), _:
            if stride_z is not None:
                logger.warning(
                    "Ignoring parameter `stride_z` for 2D data in SWITi stride "
                    "calculation."
                )
            stride_xy = int(np.ceil(crop_size_xy / np.sqrt(target_mmse_count)))
            stride = (stride_xy, stride_xy)
        case (_, _, _), _ if stride_z is not None:
            crop_size_z = patch_size[0] - overlap[0]
            coverage_z = int(np.ceil(crop_size_z / stride_z))
            coverage_remaining = target_mmse_count / coverage_z
            stride_xy = int(np.ceil(crop_size_xy / np.sqrt(coverage_remaining)))
            stride = (stride_z, stride_xy, stride_xy)
        case (_, _, _), None:
            crop_size_z = patch_size[0] - overlap[0]
            stride_z, stride_xy = _search_3D_strides(
                crop_size_xy, crop_size_z, target_mmse_count
            )
            stride = (stride_z, stride_xy, stride_xy)
        case _:
            raise ValueError(
                f"Invalid `patch_size`, only 2D or 3D is allowed, got {patch_size}."
            )

    achieved_mmse_count = np.prod(
        [
            effective_mmse_count(ps, s, o)
            for ps, s, o in zip(patch_size, stride, overlap, strict=True)
        ]
    ).item()
    return stride, achieved_mmse_count


def _search_3D_strides(
    crop_size_xy: int, crop_size_z: int, target_mmse_count: int
) -> tuple[int, int]:
    """
    Find the stride that results in pixel coverage closest to the `target_mmse_count`.

    The stride is also subject to the following constraints:
    - The effective MMSE will always be greater than or equal to `target_mmse_count`,
    - The strides in X and Y are equal, and
    - If there are two solutions that result in the same optimum effective MMSE count
      then the result which minimizes the absolute difference between the stride in XY
      and the stride in Z is chosen.

    Parameters
    ----------
    crop_size_xy : int
        The size of the cropped tile in x and y after dropping the margins.
    crop_size_z : int
        The size of the cropped tile in z after dropping the margins.
    target_mmse_count : int
        The desired pixel coverage.

    Returns
    -------
    int
        The stride in XY.
    int
        The stride in Z.
    """

    def objective(value: tuple[int, int]) -> float:
        """Objective to minimize.

        Parameters
        ----------
        value : tuple[int, int]
            First value is the stride in the XY dimensions, second value is the stride
            in the Z dimension.

        Returns
        -------
        int
            Evaluated objective function.
        """
        stride_xy, stride_z = value
        coverage_xy = int(np.ceil(crop_size_xy / stride_xy))
        coverage_z = int(np.ceil(crop_size_z / stride_z))

        primary_objective = coverage_xy**2 * coverage_z - target_mmse_count

        # must evaluate to greater than the target count
        if primary_objective < 0:
            return np.inf

        # secondary objective:
        # in the case of a tie, choose solution that minimizes abs(stride_z - stride_xy)
        secondary_objective = abs(stride_z - stride_xy)

        # Larger than the maximum possible abs(stride_z - stride_xy)
        tie_break_scale = max(crop_size_xy, crop_size_z) + 1

        return primary_objective * tie_break_scale + secondary_objective

    result = brute(
        objective,
        ranges=(
            # stride cannot be greater than half the crop size,
            # as it would result in uneven coverage
            slice(1, max(crop_size_xy // 2, 1) + 1),
            slice(1, max(crop_size_z // 2, 1) + 1),
        ),
        finish=None,
    )

    stride_xy, stride_z = result.astype(int).tolist()
    return stride_z, stride_xy

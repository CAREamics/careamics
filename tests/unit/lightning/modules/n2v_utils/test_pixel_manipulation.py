import itertools

import numpy as np
import pytest
import torch

from careamics.lightning.modules.n2v_utils.pixel_manipulation import (
    _apply_struct_mask,
    _build_struct_pattern,
    _create_neg_center_pixel_mask,
    _create_neg_struct_mask,
)
from careamics.lightning.modules.n2v_utils.struct_mask_parameters import (
    StructMaskParameters,
)

# --- Utils

COORDS_2D = torch.tensor([[0, 5, 8], [0, 21, 16]])
COORDS_3D = torch.tensor([[0, 7, 5, 8], [0, 2, 21, 16]])
COORDS_2D_BATCH = torch.tensor([[0, 5, 8], [0, 21, 16], [1, 12, 5]])
COORDS_3D_BATCH = torch.tensor([[0, 7, 5, 8], [0, 2, 21, 16], [1, 5, 12, 5]])

AXIS = [0, 1, 2]
SPAN = [3, 5, 7]

# --- Unit tests


@pytest.mark.parametrize("axis, span", list(itertools.product(AXIS, SPAN)))
def test_build_struct_pattern(axis, span):
    """Test that the struct pattern is built correctly."""
    expected_n_pixels = span - 1 if axis in [0, 1] else 2 * (span - 1)
    expected_n_dims = 1 if axis in [0, 1] else 2

    # create mask
    mask = _build_struct_pattern(span=span, axis=axis, device="cpu")

    # x and y coordinates of non-zero values in the mask
    ys, xs = torch.where(mask == 1)
    n_unique_ys = len(torch.unique(ys))
    n_unique_xs = len(torch.unique(xs))
    n_dims = (n_unique_ys > 1) + (n_unique_xs > 1)

    assert len(ys) == expected_n_pixels
    assert n_dims == expected_n_dims


@pytest.mark.parametrize(
    "ndims",
    [
        3,  # CYX
        4,  # CZYX
    ],
)
def test_create_neg_center_pixel_mask(ndims):
    """Test that the central pixel is correctly excluded."""
    subpatch_size = 11
    center_idx = subpatch_size // 2

    # get mask
    mask = _create_neg_center_pixel_mask(ndims, subpatch_size, device="cpu")

    assert not mask[(center_idx,) * ndims]
    assert mask.sum() == subpatch_size**ndims - 1


@pytest.mark.parametrize(
    "ndims, axis, span",
    list(
        itertools.product(
            [3, 4],  # BYX or BZYX
            AXIS,
            SPAN,
        )
    ),
)
def test_create_neg_struct_mask(ndims, axis, span):
    """Test that structN2V pattern is correctly excluded."""
    subpatch_size = 11
    expected_n_pixels = span if axis in [0, 1] else 2 * span - 1  # with center pixel
    expected_n_dims = 1 if axis in [0, 1] else 2

    # get mask
    mask = _create_neg_struct_mask(
        ndims, subpatch_size, StructMaskParameters(axis=axis, span=span), device="cpu"
    )

    # coordinates of non-zero values in the mask
    coords = torch.where(mask == 0)
    n_unique_xs = len(torch.unique(coords[-1]))
    n_unique_ys = len(torch.unique(coords[-2]))
    n_dims = (n_unique_ys > 1) + (n_unique_xs > 1)

    assert mask.sum() == subpatch_size**ndims - expected_n_pixels
    assert n_dims == expected_n_dims


@pytest.mark.parametrize(
    "coords, axis, span",
    list(
        itertools.product(
            # coords
            [COORDS_2D, COORDS_3D, COORDS_2D_BATCH, COORDS_3D_BATCH],
            # axis and span of the structN2V mask
            AXIS,
            SPAN,
        )
    ),
)
def test_apply_struct_mask(coords, axis, span):
    """Test that structN2V mask is correctly applied to the coordinates."""
    npts = coords.shape[0]
    ndims = coords.shape[1]
    nbatch = coords[:, 0].max().item() + 1
    shape = (nbatch, 32, 32) if ndims == 3 else (nbatch, 8, 32, 32)
    patch = torch.tensor(np.arange(np.prod(shape)).reshape(shape).astype(np.float32))

    expected_n_pixels = npts * (span - 1) if axis in [0, 1] else npts * (2 * (span - 1))
    expected_n_dims = 1 if axis in [0, 1] else 2

    masked_patch = _apply_struct_mask(
        patch.clone(), coords, StructMaskParameters(axis=axis, span=span)
    )

    diffs = torch.where(masked_patch != patch)
    n_unique_ys = len(torch.unique(diffs[-2]))
    n_unique_xs = len(torch.unique(diffs[-1]))
    n_dims = (n_unique_ys > npts) + (n_unique_xs > npts)

    assert len(diffs[0]) == expected_n_pixels
    assert n_dims == expected_n_dims

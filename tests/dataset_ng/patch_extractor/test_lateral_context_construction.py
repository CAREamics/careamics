from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from skimage.transform import resize

from careamics.dataset_ng.patch_extractor.image_stack import InMemoryImageStack
from careamics.dataset_ng.patch_extractor.patch_construction import (
    lateral_context_patch_constr,
)


def _assert_lc_centralized(lc_input: NDArray[Any]):
    multiscale_count = lc_input.shape[0]
    n_channels = lc_input.shape[1]
    patch_size = lc_input.shape[2:]

    primary_patch = lc_input[0]
    for scale in range(1, multiscale_count):
        lc_patch = lc_input[scale]

        scale_factor = 2**scale
        equiv_size = tuple(ps // scale_factor for ps in patch_size)

        scaled = resize(primary_patch, (n_channels, *equiv_size))

        central_patch = lc_patch[
            :,
            *(
                slice(ps // 2 - es // 2, ps // 2 + es // 2, None)
                for ps, es in zip(patch_size, equiv_size, strict=True)
            ),
        ]

        border_crop = (..., *(slice(2, -2, None) for _ in patch_size))
        # there are some border differences since resize won't interpolate the same way
        np.testing.assert_allclose(scaled[border_crop], central_patch[border_crop])


@pytest.mark.parametrize(
    ["data_shape", "patch_size", "axes"],
    [
        ((512, 496), (64, 64), "YX"),
        ((451, 501, 2), (64, 64), "YXC"),
        ((512, 512, 64), (32, 64, 64), "YXZ"),
        ((2, 512, 496, 65), (32, 64, 64), "CYXZ"),
    ],
)
@pytest.mark.parametrize("multiscale_count", [2, 3, 4])
def test_lateral_context_constructor(
    data_shape: tuple[int, ...],
    patch_size: tuple[int, ...],
    axes: str,
    multiscale_count: int,
):
    rng = np.random.default_rng(seed=42)
    data = rng.random(data_shape)
    image_stack = InMemoryImageStack.from_array(data, axes)

    constructor_func = lateral_context_patch_constr(multiscale_count, "reflect")
    coords = [tuple(0 for _ in patch_size), tuple(ps // 2 for ps in (patch_size))]
    for coord in coords:
        lc_input = constructor_func(image_stack, 0, coord, patch_size)
        _assert_lc_centralized(lc_input)

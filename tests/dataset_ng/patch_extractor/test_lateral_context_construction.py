from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from skimage.transform import resize

from careamics.dataset_ng.image_stack import InMemoryImageStack
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patch_extractor.patch_construction import (
    lateral_context_patch_constr,
)
from careamics.dataset_ng.patching_strategies import RandomPatchingStrategy


def _assert_lc_centralized(lc_patch: NDArray[Any]):
    """
    Assert that the central region of each lateral context patch contains the primary
    patch (but each at a smaller scale).

    Parameters
    ----------
    lc_input : NDArray[Any]
        The lateral context input with the dimensions CL(Z)YX, where L is the lateral
        context inputs. The first patch in L is the primary patch at the original
        image scale.
    """
    multiscale_count = lc_patch.shape[1]
    n_channels = lc_patch.shape[0]
    patch_size = lc_patch.shape[2:]

    primary_patch = lc_patch[:, 0, ...]
    for scale in range(1, multiscale_count):
        lc_scale = lc_patch[:, scale, ...]

        scale_factor = 2**scale
        # size of the data in the primary patch at this scale
        equiv_size = tuple(ps // scale_factor for ps in patch_size)

        # the primary patch scaled to the size of the data in the lc patch
        scaled = resize(primary_patch, (n_channels, *equiv_size))
        # the centre of the lc, that should contain the data from the primary input
        central_region = lc_scale[
            :,
            *(
                slice(ps // 2 - es // 2, ps // 2 + es // 2, None)
                for ps, es in zip(patch_size, equiv_size, strict=True)
            ),
        ]

        # there are some border differences since resize won't interpolate the same way
        border_crop = (..., *(slice(2, -2, None) for _ in patch_size))
        # assert that the scaled primary patch is the same as the lc central region
        np.testing.assert_allclose(scaled[border_crop], central_region[border_crop])


@pytest.mark.parametrize(
    ["data_shape", "patch_size", "axes"],
    [
        ((512, 496), (64, 64), "YX"),
        ((451, 501, 2), (64, 64), "YXC"),
        ((512, 512, 128), (32, 64, 64), "YXZ"),
        ((2, 512, 497, 129), (32, 64, 64), "CYXZ"),
    ],
)
@pytest.mark.parametrize("channels", [[0], None])
def test_lateral_context_constructor(
    data_shape: tuple[int, ...],
    patch_size: tuple[int, ...],
    channels: int | None,
    axes: str,
):
    """Test the lateral context patch constructor function."""
    rng = np.random.default_rng(seed=42)
    multiscale_count = 4
    data = rng.random(data_shape)
    image_stack = InMemoryImageStack.from_array(data, axes)

    constructor_func = lateral_context_patch_constr(multiscale_count, "reflect")
    # test coord at edge (which will have padded lc) and coord at centre
    coords = [tuple(0 for _ in patch_size), tuple(ps // 2 for ps in (patch_size))]
    for coord in coords:
        lc_patch = constructor_func(image_stack, 0, channels, coord, patch_size)
        assert lc_patch.shape[1] == multiscale_count
        _assert_lc_centralized(lc_patch)


def test_patch_extractor_lc_injection():
    rng = np.random.default_rng(seed=42)
    multiscale_count = 4
    axes = "SYX"
    data_shapes = [(1, 512, 496), (3, 451, 501)]
    # create data
    image_stacks = [
        InMemoryImageStack.from_array(rng.random(data_shape), axes)
        for data_shape in data_shapes
    ]
    # inject patch extractor with constructor func
    constructor_func = lateral_context_patch_constr(multiscale_count, "reflect")
    patch_extractor = PatchExtractor(image_stacks, constructor_func)

    # use random patching strategy to generate patch specs and extract lc patches
    patch_size = (64, 64)
    patching_strat = RandomPatchingStrategy(patch_extractor.shapes, patch_size, seed=42)
    for idx in range(patching_strat.n_patches):
        patch_spec = patching_strat.get_patch_spec(idx)
        lc_patch = patch_extractor.extract_patch(**patch_spec)
        assert lc_patch.shape[1] == multiscale_count
        _assert_lc_centralized(lc_patch)


def test_lateral_context_constructor_with_channels():
    """Test that the lateral context constructor raises an error with multiple
    channels."""
    rng = np.random.default_rng(seed=42)
    data = rng.random((2, 512, 496))
    image_stack = InMemoryImageStack.from_array(data, "CYX")

    constructor_func = lateral_context_patch_constr(4, "reflect")
    with pytest.raises(NotImplementedError):
        constructor_func(image_stack, 0, [0, 1], (0, 0), (64, 64))

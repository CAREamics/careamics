from typing import Any

import numpy as np
from numpy.typing import NDArray

from .patch_extractor import PatchExtractor
from .patching_strategies import PatchingStrategy


def create_uncorrelated_channel_patch(
    idx: int,
    patch_extractor: PatchExtractor,
    patching_strategy: PatchingStrategy,
    alphas: list[float],
    rng: np.random.Generator | None,
) -> NDArray[Any]:
    if rng is None:
        rng = np.random.default_rng()

    n_channels = len(alphas)

    # in the original dataset, new random indices are chosen for each channel
    # the other channels can come from anywhere in the entire dataset
    indices = (idx, *rng.integers(patching_strategy.n_patches, size=(n_channels - 1)))

    patch_specs = [patching_strategy.get_patch_spec(i) for i in indices]
    # dimensions C(L)(Z)YX - there may or may not be an L (lateral context) dimension
    patches = np.concat(
        [
            patch_extractor.extract_channel_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                channel_idx=c,
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )
            for c, patch_spec in enumerate(patch_specs)
        ],
        axis=0,
    )

    ndims = len(patches.shape) - 1
    alpha_broadcast = np.array(alphas)[:, *(np.newaxis for _ in range(ndims))]
    return (alpha_broadcast * patches).sum(axis=0)

"""Filter patch using a maximum filter."""

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import maximum_filter
from tqdm import tqdm

from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
)
from careamics.dataset_ng.patch_filter.patch_filter_protocol import PatchFilterProtocol
from careamics.dataset_ng.patching_strategies import TilingStrategy


class MaxPatchFilter(PatchFilterProtocol):
    """
    A patch filter based on thresholding the maximum filter of the patch.

    Inspired by the CSBDeep approach.

    Attributes
    ----------
    threshold : float
        Threshold for the maximum filter of the patch.
    p : float
        Probability of applying the filter to a patch.
    rng : np.random.Generator
        Random number generator for stochastic filtering.
    """

    def __init__(
        self,
        threshold: float,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:

        self.threshold = threshold

        self.p = p
        self.rng = np.random.default_rng(seed)

    def filter_out(self, patch: np.ndarray) -> bool:
        if self.rng.uniform(0, 1) < self.p:

            patch_shape = [(p // 2 if p > 1 else 1) for p in patch.shape]
            filtered = maximum_filter(patch, patch_shape, mode="constant")

            return (filtered < self.threshold).any()
        return False

    @staticmethod
    def filter_map(
        image: np.ndarray,
        patch_size: Sequence[int],
    ) -> np.ndarray:
        """
        Compute the maximum map of an image.

        The map is computed over non-overlapping patches. This method can be used
        to assess a useful threshold for the MaxPatchFilter filter.

        Parameters
        ----------
        image : numpy.NDArray
            The image for which to compute the map, must be 2D or 3D.
        patch_size : Sequence[int]
            The size of the patches to compute the map over. Must be a sequence
            of two integers.

        Returns
        -------
        numpy.NDArray
            The max map of the patch.

        Raises
        ------
        ValueError
            If the image is not 2D or 3D.

        Example
        -------
        The `filter_map` method can be used to assess a useful threshold for the
        Shannon entropy filter. Below is an example of how to compute and visualize
        the Shannon entropy map of a random image and visualize thresholded versions
        of the map.
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from careamics.dataset_ng.patch_filter import MaxPatchFilter
        >>> rng = np.random.default_rng(42)
        >>> image = rng.binomial(20, 0.1, (256, 256)).astype(np.float32)
        >>> image[64:192, 64:192] += rng.normal(50, 5, (128, 128))
        >>> image[96:160, 96:160] = rng.poisson(image[96:160, 96:160])
        >>> patch_size = (16, 16)
        >>> max_filtered = MaxPatchFilter.filter_map(image, patch_size)
        >>> fig, ax = plt.subplots(1, 5, figsize=(20, 5))
        >>> for i, thresh in enumerate([50 + i*5 for i in range(5)]):
        ... ax[i].imshow(max_filtered > thresh, cmap="gray")
        ... ax[i].set_title(f"Threshold: {thresh}")
        >>> plt.show()
        """
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError("Image must be 2D or 3D.")

        axes = "YX" if len(patch_size) == 2 else "ZYX"

        max_filtered = np.zeros_like(image, dtype=float)

        extractor = create_array_extractor(source=[image], axes=axes)
        tiling = TilingStrategy(
            data_shapes=[(1, 1, *image.shape)],
            tile_size=patch_size,
            overlaps=(0,) * len(patch_size),  # no overlap
        )
        max_patch_size = [p // 2 for p in patch_size]

        for idx in tqdm(range(tiling.n_patches), desc="Computing max map"):
            patch_spec = tiling.get_patch_spec(idx)
            patch = extractor.extract_patch(
                data_idx=0,
                sample_idx=0,
                coords=patch_spec["coords"],
                patch_size=patch_size,
            )

            coordinates = tuple(
                slice(patch_spec["coords"][i], patch_spec["coords"][i] + p)
                for i, p in enumerate(patch_size)
            )
            max_filtered[coordinates] = maximum_filter(
                patch.squeeze(), max_patch_size, mode="constant"
            )

        return max_filtered

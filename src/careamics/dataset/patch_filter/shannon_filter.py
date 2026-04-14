"""Filter patches based on Shannon entropy threshold."""

from collections.abc import Sequence

import numpy as np
from skimage.measure import shannon_entropy
from tqdm import tqdm

from careamics.dataset.image_stack_loader import load_arrays
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patch_filter.patch_filter_protocol import PatchFilterProtocol
from careamics.dataset.patching_strategies import TilingStrategy


class ShannonPatchFilter(PatchFilterProtocol):
    """Filter patches based on Shannon entropy threshold.

    Parameters
    ----------
    threshold : float
        Shannon entropy threshold; patches below are filtered out.

    Attributes
    ----------
    threshold : float
        Threshold for the Shannon entropy of the patch.
    """

    def __init__(self, threshold: float) -> None:
        """Create a ShannonPatchFilter.

        This filter removes patches whose Shannon entropy is below a specified
        threshold.

        Parameters
        ----------
        threshold : float
            Threshold for the Shannon entropy of the patch.

        Raises
        ------
        ValueError
            If threshold is negative.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative.")

        self.threshold = threshold

    def filter_out(self, patch: np.ndarray) -> bool:
        """
        Determine whether to filter out a patch based on its Shannon entropy.

        Parameters
        ----------
        patch : numpy.NDArray
            The patch to evaluate.

        Returns
        -------
        bool
            True if the patch should be filtered out, False otherwise.
        """
        return shannon_entropy(patch) < self.threshold

    @staticmethod
    def filter_map(
        image: np.ndarray,
        patch_size: Sequence[int],
    ) -> np.ndarray:
        """
        Compute the Shannon entropy map of an image.

        The entropy is computed over non-overlapping patches. This method can be used
        to assess a useful threshold for the Shannon entropy filter.

        Parameters
        ----------
        image : numpy.NDArray
            The image for which to compute the entropy map, must be 2D or 3D.
        patch_size : Sequence[int]
            The size of the patches to compute the entropy over. Must be a sequence
            of two integers.

        Returns
        -------
        numpy.NDArray
            The Shannon entropy map of the patch.

        Raises
        ------
        ValueError
            If the image is not 2D or 3D.

        Examples
        --------
        Assess a useful threshold by computing and visualizing the entropy map:
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> from careamics.dataset.patch_filter import ShannonPatchFilter
        >>> rng = np.random.default_rng(42)
        >>> image = rng.binomial(20, 0.1, (256, 256)).astype(np.float32)
        >>> image[64:192, 64:192] += rng.normal(50, 5, (128, 128))
        >>> image[96:160, 96:160] = rng.poisson(image[96:160, 96:160])
        >>> patch_size = (16, 16)
        >>> entropy_map = ShannonPatchFilter.filter_map(image, patch_size)
        >>> fig, ax = plt.subplots(1, 5, figsize=(20, 5)) # doctest: +SKIP
        >>> for i, thresh in enumerate([2 + 1.5 * i for i in range(5)]):
        ...     ax[i].imshow(entropy_map >= thresh, cmap="gray") #doctest: +SKIP
        ...     ax[i].set_title(f"Threshold: {thresh}") #doctest: +SKIP
        >>> plt.show() # doctest: +SKIP
        """
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError("Image must be 2D or 3D.")

        axes = "YX" if len(patch_size) == 2 else "ZYX"

        shannon_img = np.zeros_like(image, dtype=float)

        image_stacks = load_arrays(source=[image], axes=axes)
        extractor = PatchExtractor(image_stacks)
        tiling = TilingStrategy(
            data_shapes=[(1, 1, *image.shape)],
            patch_size=patch_size,
            overlaps=(0,) * len(patch_size),  # no overlap
        )

        for idx in tqdm(range(tiling.n_patches), desc="Computing Shannon Entropy map"):
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
            shannon_img[coordinates] = shannon_entropy(patch)

        return shannon_img

    @staticmethod
    def apply_filter(
        filter_map: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Apply the Shannon entropy filter to a precomputed filter map.

        The filter map is the output of the `filter_map` method.

        Parameters
        ----------
        filter_map : numpy.NDArray
            The precomputed Shannon entropy map of the image.
        threshold : float
            The Shannon entropy threshold for filtering.

        Returns
        -------
        numpy.NDArray
            A boolean array where True indicates that the patch should be kept
            (not filtered out) and False indicates that the patch should be filtered
            out.
        """
        return filter_map >= threshold

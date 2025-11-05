from collections.abc import Sequence
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from .image_stack import ImageStack


class PatchConstructor(Protocol):

    def __call__(
        self,
        image_stack: ImageStack,
        sample_idx: int,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray[Any]: ...


def basic_patch_constr(
    image_stack: ImageStack,
    sample_idx: int,
    coords: Sequence[int],
    patch_size: Sequence[int],
) -> NDArray[Any]:
    return image_stack.extract_patch(
        sample_idx=sample_idx, coords=coords, patch_size=patch_size
    )


def lateral_context_patch_constr(
    multiscale_count: int, padding_mode: Literal["reflect", "wrap"]
) -> PatchConstructor:
    def constructor_func(
        image_stack: ImageStack,
        sample_idx: int,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray[Any]:
        shape = image_stack.data_shape
        spatial_shape = shape[2:]
        n_channels = shape[1]

        patch = np.zeros((multiscale_count, n_channels, *patch_size))
        for scale in range(multiscale_count):
            lc_patch_size = np.array(patch_size) * (2**scale)
            lc_start = np.array(coords) + np.array(patch_size) // 2 - lc_patch_size // 2
            lc_end = lc_start + np.array(lc_patch_size)

            start_clipped = np.clip(
                lc_start, np.zeros_like(spatial_shape), np.array(spatial_shape)
            )
            end_clipped = np.clip(
                lc_end, np.zeros_like(spatial_shape), np.array(spatial_shape)
            )
            size_clipped = end_clipped - start_clipped

            lc_patch = image_stack.extract_patch(
                sample_idx, start_clipped, size_clipped
            )
            pad_before = start_clipped - lc_start
            pad_after = lc_end - end_clipped
            pad_width = np.concat(
                [
                    np.zeros((1, 2), dtype=int),
                    np.stack([pad_before, pad_after], axis=-1),
                ]
            )
            lc_patch = np.pad(
                # zeros to not pad the channel axis
                lc_patch,
                pad_width,
                mode=padding_mode,
            )
            lc_patch = resize(lc_patch, (n_channels, *patch_size))
            patch[scale] = lc_patch
        return patch

    return constructor_func

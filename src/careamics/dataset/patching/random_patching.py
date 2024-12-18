"""Random patching utilities."""

from collections.abc import Generator
from typing import Optional, Union

import numpy as np
import zarr

from .validate_patch_dimension import validate_patch_dimensions


# TOOD split in testable functions
def extract_patches_random(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    target: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Generator[tuple[np.ndarray, Optional[np.ndarray]], None, None]:
    """
    Generate patches from an array in a random manner.

    The method calculates how many patches the image can be divided into and then
    extracts an equal number of random patches.

    It returns a generator that yields the following:

    - patch: np.ndarray, dimension C(Z)YX.
    - target_patch: np.ndarray, dimension C(Z)YX, if the target is present, None
        otherwise.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    patch_size : tuple of int
        Patch sizes in each dimension.
    target : Optional[np.ndarray], optional
        Target array, by default None.
    seed : int or None, default=None
        Random seed.

    Yields
    ------
    Generator[np.ndarray, None, None]
        Generator of patches.
    """
    rng = np.random.default_rng(seed=seed)

    is_3d_patch = len(patch_size) == 3

    # patches sanity check
    validate_patch_dimensions(arr, patch_size, is_3d_patch)

    # Update patch size to encompass S and C dimensions
    patch_size = [1, arr.shape[1], *patch_size]

    # iterate over the number of samples (S or T)
    for sample_idx in range(arr.shape[0]):
        # get sample array
        sample: np.ndarray = arr[sample_idx, ...]

        # same for target
        if target is not None:
            target_sample: np.ndarray = target[sample_idx, ...]

        # calculate the number of patches
        n_patches = np.ceil(np.prod(sample.shape) / np.prod(patch_size)).astype(int)

        # iterate over the number of patches
        for _ in range(n_patches):
            # get crop coordinates
            crop_coords = [
                rng.integers(0, sample.shape[i] - patch_size[1:][i], endpoint=True)
                for i in range(len(patch_size[1:]))
            ]

            # extract patch
            patch = (
                sample[
                    (
                        ...,  # type: ignore
                        *[  # type: ignore
                            slice(c, c + patch_size[1:][i])
                            for i, c in enumerate(crop_coords)
                        ],
                    )
                ]
                .copy()
                .astype(np.float32)
            )

            # same for target
            if target is not None:
                target_patch = (
                    target_sample[
                        (
                            ...,  # type: ignore
                            *[  # type: ignore
                                slice(c, c + patch_size[1:][i])
                                for i, c in enumerate(crop_coords)
                            ],
                        )
                    ]
                    .copy()
                    .astype(np.float32)
                )
                # return patch and target patch
                yield patch, target_patch
            else:
                # return patch
                yield patch, None


def extract_patches_random_from_chunks(
    arr: zarr.Array,
    patch_size: Union[list[int], tuple[int, ...]],
    chunk_size: Union[list[int], tuple[int, ...]],
    chunk_limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generate patches from an array in a random manner.

    The method calculates how many patches the image can be divided into and then
    extracts an equal number of random patches.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    patch_size : Union[list[int], tuple[int, ...]]
        Patch sizes in each dimension.
    chunk_size : Union[list[int], tuple[int, ...]]
        Chunk sizes to load from the.
    chunk_limit : Optional[int], optional
        Number of chunks to load, by default None.
    seed : Optional[int], optional
        Random seed, by default None.

    Yields
    ------
    Generator[np.ndarray, None, None]
        Generator of patches.
    """
    is_3d_patch = len(patch_size) == 3

    # Patches sanity check
    validate_patch_dimensions(arr, patch_size, is_3d_patch)

    rng = np.random.default_rng(seed=seed)
    num_chunks = chunk_limit if chunk_limit else np.prod(arr._cdata_shape)

    # Iterate over num chunks in the array
    for _ in range(num_chunks):
        chunk_crop_coords = [
            rng.integers(0, max(0, arr.shape[i] - chunk_size[i]), endpoint=True)
            for i in range(len(chunk_size))
        ]
        chunk = arr[
            (
                ...,
                *[slice(c, c + chunk_size[i]) for i, c in enumerate(chunk_crop_coords)],
            )
        ].squeeze()

        # Add a singleton dimension if the chunk does not have a sample dimension
        if len(chunk.shape) == len(patch_size):
            chunk = np.expand_dims(chunk, axis=0)

        # Iterate over num samples (S)
        for sample_idx in range(chunk.shape[0]):
            spatial_chunk = chunk[sample_idx]
            assert len(spatial_chunk.shape) == len(
                patch_size
            ), "Requested chunk shape is not equal to patch size"

            n_patches = np.ceil(
                np.prod(spatial_chunk.shape) / np.prod(patch_size)
            ).astype(int)

            # Iterate over the number of patches
            for _ in range(n_patches):
                patch_crop_coords = [
                    rng.integers(
                        0, spatial_chunk.shape[i] - patch_size[i], endpoint=True
                    )
                    for i in range(len(patch_size))
                ]
                patch = (
                    spatial_chunk[
                        (
                            ...,
                            *[
                                slice(c, c + patch_size[i])
                                for i, c in enumerate(patch_crop_coords)
                            ],
                        )
                    ]
                    .copy()
                    .astype(np.float32)
                )
                yield patch

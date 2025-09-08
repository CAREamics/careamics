"""Sequential patching functions."""

from typing import Union

import numpy as np
from skimage.util import view_as_windows

from .validate_patch_dimension import validate_patch_dimensions

def extract_patches_sequential(
    arr: np.ndarray, 
    patch_size: Union[list[int], tuple[int, ...]], 
    target: np.ndarray = None,
    axes: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract patches from a single image sequentially with support for 1D, 2D, and 3D data.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    patch_size : Union[list[int], tuple[int, ...]]
        Patch size.
    target : np.ndarray, optional
        Target array, by default None.
    axes : str, optional
        Axes string describing array dimensions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Patches and targets (if provided).
    """
    # Convert patch_size to list for consistency
    if isinstance(patch_size, tuple):
        patch_size = list(patch_size)
    
    # Determine if this is 3D based on patch_size length (legacy behavior)
    is_3d_patch = len(patch_size) == 3
    
    # Validate patch dimensions with axes support
    validate_patch_dimensions(arr, patch_size, is_3d_patch, axes)
    
    # Determine spatial dimensions
    if axes is not None:
        spatial_axes = [ax for ax in axes if ax in 'XYZ']
        n_spatial_dims = len(spatial_axes)
    else:
        # Infer from patch_size length and array shape
        if len(patch_size) == 1 and len(arr.shape) == 2:
            n_spatial_dims = 1  # 1D case: (samples, X)
        else:
            n_spatial_dims = len(patch_size)
    
    # Route to appropriate patching function
    if n_spatial_dims == 1:
        return _extract_patches_1d(arr, patch_size[0], target)
    elif n_spatial_dims == 2:
        return _extract_patches_2d(arr, patch_size, target)
    elif n_spatial_dims == 3:
        return _extract_patches_3d(arr, patch_size, target)
    else:
        raise ValueError(f"Unsupported number of spatial dimensions: {n_spatial_dims}")


def _compute_number_of_patches(
    arr_shape: tuple[int, ...], patch_sizes: Union[list[int], tuple[int, ...]]
) -> tuple[int, ...]:
    """
    Compute the number of patches that fit in each dimension.

    Parameters
    ----------
    arr_shape : tuple[int, ...]
        Shape of the input array.
    patch_sizes : Union[list[int], tuple[int, ...]
        Shape of the patches.

    Returns
    -------
    tuple[int, ...]
        Number of patches in each dimension.
    """
    if len(arr_shape) != len(patch_sizes):
        raise ValueError(
            f"Array shape {arr_shape} and patch size {patch_sizes} should have the "
            f"same dimension, including singleton dimension for S and equal dimension "
            f"for C."
        )

    try:
        n_patches = [
            np.ceil(arr_shape[i] / patch_sizes[i]).astype(int)
            for i in range(len(patch_sizes))
        ]
    except IndexError as e:
        raise ValueError(
            f"Patch size {patch_sizes} is not compatible with array shape {arr_shape}"
        ) from e

    return tuple(n_patches)


def _compute_overlap(
    arr_shape: tuple[int, ...], patch_sizes: Union[list[int], tuple[int, ...]]
) -> tuple[int, ...]:
    """
    Compute the overlap between patches in each dimension.

    If the array dimensions are divisible by the patch sizes, then the overlap is
    0. Otherwise, it is the result of the division rounded to the upper value.

    Parameters
    ----------
    arr_shape : tuple[int, ...]
        Input array shape.
    patch_sizes : Union[list[int], tuple[int, ...]]
        Size of the patches.

    Returns
    -------
    tuple[int, ...]
        Overlap between patches in each dimension.
    """
    n_patches = _compute_number_of_patches(arr_shape, patch_sizes)

    overlap = [
        np.ceil(
            np.clip(n_patches[i] * patch_sizes[i] - arr_shape[i], 0, None)
            / max(1, (n_patches[i] - 1))
        ).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(overlap)


def _compute_patch_steps(
    patch_sizes: Union[list[int], tuple[int, ...]], overlaps: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Compute steps between patches.

    Parameters
    ----------
    patch_sizes : tuple[int]
        Size of the patches.
    overlaps : tuple[int]
        Overlap between patches.

    Returns
    -------
    tuple[int]
        Steps between patches.
    """
    steps = [
        min(patch_sizes[i] - overlaps[i], patch_sizes[i])
        for i in range(len(patch_sizes))
    ]
    return tuple(steps)


# TODO why stack the target here and not on a different dimension before this function?
def _compute_patch_views(
    arr: np.ndarray,
    window_shape: list[int],
    step: tuple[int, ...],
    output_shape: list[int],
    target: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute views of an array corresponding to patches.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the views are extracted.
    window_shape : tuple[int]
        Shape of the views.
    step : tuple[int]
        Steps between views.
    output_shape : tuple[int]
        Shape of the output array.
    target : Optional[np.ndarray], optional
        Target array, by default None.

    Returns
    -------
    np.ndarray
        Array with views dimension.
    """
    rng = np.random.default_rng()

    if target is not None:
        arr = np.stack([arr, target], axis=0)
        window_shape = [arr.shape[0], *window_shape]
        step = (arr.shape[0], *step)
        output_shape = [-1, arr.shape[0], arr.shape[2], *output_shape[2:]]

    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    rng.shuffle(patches, axis=0)
    return patches

def _extract_patches_1d(
    arr: np.ndarray, 
    patch_size: int, 
    target: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D patches sequentially.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (samples, length).
    patch_size : int
        Size of patches along spatial dimension.
    target : np.ndarray, optional
        Target array with same shape as arr.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Patches and target patches.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for 1D patching (samples, length), got {arr.ndim}D")
    
    n_samples, length = arr.shape
    
    if patch_size > length:
        raise ValueError(f"Patch size {patch_size} exceeds sequence length {length}")
    
    # Calculate number of patches per sample
    n_patches_per_sample = length - patch_size + 1
    
    if n_patches_per_sample <= 0:
        raise ValueError(f"Cannot extract patches: patch_size={patch_size}, length={length}")
    
    # Extract patches
    patches = []
    target_patches = [] if target is not None else None
    
    for sample_idx in range(n_samples):
        for start_idx in range(n_patches_per_sample):
            # Extract patch from input
            patch = arr[sample_idx, start_idx:start_idx + patch_size]
            patches.append(patch)
            
            # Extract corresponding target patch if target provided
            if target is not None:
                target_patch = target[sample_idx, start_idx:start_idx + patch_size]
                target_patches.append(target_patch)
    
    patches_array = np.array(patches)
    target_patches_array = np.array(target_patches) if target_patches is not None else None
    
    return patches_array, target_patches_array

def _extract_patches_2d(
    arr: np.ndarray, 
    patch_size: Union[list[int], tuple[int, ...]], 
    target: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 2D patches sequentially.
    
    This function needs to be implemented based on the existing 2D logic.
    For now, it's a placeholder that raises NotImplementedError.
    """
    raise NotImplementedError("2D sequential patching implementation needed")


def _extract_patches_3d(
    arr: np.ndarray, 
    patch_size: Union[list[int], tuple[int, ...]], 
    target: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D patches sequentially.
    
    This function needs to be implemented based on the existing 3D logic.
    For now, it's a placeholder that raises NotImplementedError.
    """
    raise NotImplementedError("3D sequential patching implementation needed")

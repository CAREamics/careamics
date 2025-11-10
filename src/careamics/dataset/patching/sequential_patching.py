"""Sequential patching functions."""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from skimage.util import view_as_windows

from .validate_patch_dimension import validate_patch_dimensions
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def extract_patches_sequential(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    target: np.ndarray = None,
    axes: str = None,
) -> tuple[np.ndarray, np.ndarray | None]:
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
    tuple[np.ndarray, np.ndarray | None]
        Patches and targets (if provided) as numpy arrays.
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
        spatial_axes = [ax for ax in axes if ax in "XYZ"]
        n_spatial_dims = len(spatial_axes)
    else:
        # Infer from patch_size length and array shape
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
    arr: NDArray, patch_size: int, target: NDArray | None = None
) -> tuple[NDArray, NDArray | None]:
    """
    Extract 1D patches sequentially.

    Parameters
    ----------
    arr : NDArray
        Input array with shape (samples, channels, length) for 1D data after reshaping.
    patch_size : int
        Size of patches to extract.
    target : NDArray | None
        Target array, by default None.

    Returns
    -------
    tuple[NDArray, NDArray | None]
        Patches and target patches as numpy arrays.
    """
    # reshape 1D data to support convention

    # Handle both 2D (samples, length) and 3D (samples, channels, length) arrays
    # Normalise input shape
    if arr.ndim == 2:
        # Legacy format: (samples, length) -> add channel dimension
        arr = arr[:, np.newaxis, :]
        if target is not None:
            target = target[:, np.newaxis, :]
    elif arr.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D array for 1D patching, got {arr.ndim}D with shape {arr.shape}"
        )

    n_samples, n_channels, length = arr.shape

    if patch_size > length:
        raise ValueError(
            f"Patch size ({patch_size}) cannot be larger than sequence length ({length})"
        )
    full_patch_size = [1, n_channels, patch_size]
    overlaps = _compute_overlap(arr_shape=arr.shape, patch_sizes=full_patch_size)
    window_steps = _compute_patch_steps(patch_sizes=full_patch_size, overlaps=overlaps)
    output_shape = [-1, n_channels, patch_size]
    patches = _compute_patch_views(
        arr,
        window_shape=full_patch_size,
        step=window_steps,
        output_shape=output_shape,
        target=target,
    )

    # patches_list = []
    # target_patches_list = [] if target is not None else None

    # Handle target array reshaping if provided
    if target is not None:
        # target was concatenated to patches in _compute_reshaped_view
        return (
            patches[:, 0, ...],
            patches[:, 1, ...],
        )  # TODO  in _compute_reshaped_view?
    else:
        return patches, None

    # # Extract patches from each sample
    # for sample_idx in range(n_samples):
    #     sample = arr[sample_idx]  # Shape: (channels, length)

    #     # Calculate number of patches that can be extracted
    #     n_patches = length - patch_size + 1

    #     # Extract all possible patches from this sample
    #     for start_idx in range(n_patches):
    #         end_idx = start_idx + patch_size
    #         patch = sample[:, start_idx:end_idx]  # Shape: (channels, patch_size)
    #         patches_list.append(patch)

    #         # Extract corresponding target patch if available
    #         if target is not None and target_patches_list is not None:
    #             target_sample = target[sample_idx]
    #             target_patch = target_sample[:, start_idx:end_idx]
    #             target_patches_list.append(target_patch)

    # # Convert lists to numpy arrays
    # patches = np.array(patches_list) if patches_list else np.array([])
    # target_patches = np.array(target_patches_list) if target_patches_list else None

    # return patches, target_patches


def _extract_patches_2d(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    target: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract 2D patches sequentially using the original algorithm.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (S, C, Y, X) for 2D data.
    patch_size : Union[list[int], tuple[int, ...]]
        Patch size [Y, X].
    target : np.ndarray, optional
        Target array, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Patches and target patches as numpy arrays.
    """
    # Update patch size to encompass S and C dimensions
    full_patch_size = [1, arr.shape[1], *patch_size]

    # Compute overlap
    overlaps = _compute_overlap(arr_shape=arr.shape, patch_sizes=full_patch_size)

    # Create view window and overlaps
    window_steps = _compute_patch_steps(patch_sizes=full_patch_size, overlaps=overlaps)

    output_shape = [-1] + full_patch_size[1:]

    # Generate patches using view_as_windows
    patches = _compute_patch_views(
        arr,
        window_shape=full_patch_size,
        step=window_steps,
        output_shape=output_shape,
        target=target,
    )

    if target is not None:
        # target was concatenated to patches in _compute_patch_views
        return patches[:, 0, ...], patches[:, 1, ...]
    else:
        return patches, None


def _extract_patches_3d(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    target: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract 3D patches sequentially using the original algorithm.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (S, C, Z, Y, X) for 3D data.
    patch_size : Union[list[int], tuple[int, ...]]
        Patch size [Z, Y, X].
    target : np.ndarray, optional
        Target array, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Patches and target patches as numpy arrays.
    """
    # Update patch size to encompass S and C dimensions
    full_patch_size = [1, arr.shape[1], *patch_size]

    # Compute overlap
    overlaps = _compute_overlap(arr_shape=arr.shape, patch_sizes=full_patch_size)

    # Create view window and overlaps
    window_steps = _compute_patch_steps(patch_sizes=full_patch_size, overlaps=overlaps)

    output_shape = [-1] + full_patch_size[1:]

    # Generate patches using view_as_windows
    patches = _compute_patch_views(
        arr,
        window_shape=full_patch_size,
        step=window_steps,
        output_shape=output_shape,
        target=target,
    )

    if target is not None:
        # target was concatenated to patches in _compute_patch_views
        return patches[:, 0, ...], patches[:, 1, ...]
    else:
        return patches, None

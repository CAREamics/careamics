"""Patch validation functions."""

from typing import Union

import numpy as np

def validate_patch_dimensions(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    is_3d_patch: bool,
    axes: str = None,
) -> None:
    """
    Check patch size and array compatibility for 1D, 2D, and 3D data.

    This method validates the patch sizes with respect to the array dimensions.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    patch_size : Union[list[int], tuple[int, ...]]
        Size of the patches along each spatial dimension.
    is_3d_patch : bool
        Whether the patch is 3D or not.
    axes : str, optional
        Axes string describing array dimensions.

    Raises
    ------
    ValueError
        If the patch size is not consistent with the array shape.
    ValueError
        If any patch size is larger than the corresponding array dimension.
    """
    # Convert to list for consistency
    if isinstance(patch_size, tuple):
        patch_size = list(patch_size)
    
    # Only print debug for the first call to avoid spam
    if not hasattr(validate_patch_dimensions, '_debug_printed'):
        print(f"DEBUG validate_patch_dimensions: arr.shape={arr.shape}, patch_size={patch_size}, axes={axes}")
        validate_patch_dimensions._debug_printed = True
    
    # For 1D spectroscopic data, find the spatial dimension
    if axes == "SX" and len(patch_size) == 1:
        # Special handling for 1D spectroscopic data
        # Find the longest dimension (should be the spectral dimension)
        spatial_shape = (max(arr.shape),)
        n_spatial_dims = 1
        
    elif axes is not None:
        # Count spatial dimensions from axes
        spatial_axes = [ax for ax in axes if ax in 'XYZ']
        n_spatial_dims = len(spatial_axes)
        
        # Get spatial shape based on axes and array dimensions
        if n_spatial_dims == 1:
            # For 1D: find the spatial dimension
            if 'X' in axes:
                # Find the X dimension - it should be the largest dimension
                max_dim_idx = np.argmax(arr.shape)
                spatial_shape = (arr.shape[max_dim_idx],)
            else:
                # Fallback: assume last dimension is spatial
                spatial_shape = arr.shape[-1:]
        elif n_spatial_dims == 2:
            # For 2D: last 2 dimensions are typically spatial
            spatial_shape = arr.shape[-2:]
        elif n_spatial_dims == 3:
            # For 3D: last 3 dimensions are typically spatial
            spatial_shape = arr.shape[-3:]
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {n_spatial_dims}")
    else:
        # Legacy behavior: infer from array shape and is_3d_patch flag
        if len(arr.shape) < 2:
            raise ValueError(
                f"Array must have at least 2 dimensions, got {len(arr.shape)}. "
                f"Array shape: {arr.shape}"
            )
        
        if is_3d_patch:
            spatial_shape = arr.shape[-3:]
            n_spatial_dims = 3
        else:
            # Handle 1D case: if patch_size has 1 element, treat as 1D
            if len(patch_size) == 1:
                # 1D spatial data: find the largest dimension
                spatial_shape = (max(arr.shape),)
                n_spatial_dims = 1
            else:
                # 2D spatial data: last 2 dimensions are spatial
                spatial_shape = arr.shape[-2:]
                n_spatial_dims = 2

    # Validate patch_size length matches spatial dimensions
    if len(patch_size) != n_spatial_dims:
        raise ValueError(
            f"There must be a patch size for each spatial dimension "
            f"(got {len(patch_size)} patch dimensions for {n_spatial_dims} spatial dims). "
            f"Array shape: {arr.shape}, spatial shape: {spatial_shape}, axes: {axes}. "
            f"Check the axes order."
        )

    # Validate each patch dimension doesn't exceed image dimension
    for i, (patch_dim, img_dim) in enumerate(zip(patch_size, spatial_shape)):
        if patch_dim > img_dim:
            # Create descriptive dimension names
            if n_spatial_dims == 1:
                dim_names = ['X']
            elif n_spatial_dims == 2:
                dim_names = ['Y', 'X']
            elif n_spatial_dims == 3:
                dim_names = ['Z', 'Y', 'X']
            else:
                dim_names = [f"dim_{j}" for j in range(n_spatial_dims)]
            
            dim_name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
            
            raise ValueError(
                f"{dim_name} patch size is inconsistent with image shape "
                f"(got {patch_dim} patch size for {img_dim} image dimension). "
                f"Spatial shape: {spatial_shape}, axes: {axes}. Check the axes order."
            )
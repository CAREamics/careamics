"""Patch validation functions."""

from typing import Union

import numpy as np


def validate_patch_dimensions(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    is_3d_patch: bool,
    axes: str | None = None,
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

    # For 1D spectroscopic data, find the spatial dimension
    if axes == "SX" and len(patch_size) == 1:
        # Special handling for 1D spectroscopic data
        spatial_shape = (max(arr.shape),)
        n_spatial_dims = 1

    elif axes is not None:
        # Count spatial dimensions from axes
        spatial_axes = [ax for ax in axes if ax in "XYZ"]
        n_spatial_dims = len(spatial_axes)

        # Get spatial shape based on axes and array dimensions
        if n_spatial_dims == 1:
            if "X" in axes:
                max_dim_idx = np.argmax(arr.shape)
                spatial_shape = (arr.shape[max_dim_idx],)
            else:
                spatial_shape = arr.shape[-1:]
        elif n_spatial_dims == 2:
            spatial_shape = arr.shape[-2:]
        elif n_spatial_dims == 3:
            spatial_shape = arr.shape[-3:]
        else:
            raise ValueError(
                f"Unsupported number of spatial dimensions: {n_spatial_dims}"
            )
    else:
        # Legacy behavior: infer from array shape and is_3d_patch flag
        if len(arr.shape) < 2:
            raise ValueError(
                f"Array must have at least 2 dimensions, got {len(arr.shape)}. "
                f"Array shape: {arr.shape}"
            )

        if is_3d_patch:
            if len(arr.shape) < 4:
                # FIX 1: Split long string
                raise ValueError(
                    f"Array must have at least 4 dimensions (C, Z, Y, X) for "
                    f"3D patching when axes are not provided. Got shape {arr.shape}."
                )
            spatial_shape = arr.shape[-3:]
            n_spatial_dims = 3
        else:
            # Handle 1D case: if patch_size has 1 element, treat as 1D
            if len(patch_size) == 1:
                spatial_shape = (max(arr.shape),)
                n_spatial_dims = 1
            else:
                # 2D spatial data
                spatial_shape = arr.shape[-2:]
                n_spatial_dims = 2

    if len(patch_size) != n_spatial_dims:
        raise ValueError(
            f"There must be a patch size for each spatial dimension "
            f"(got {len(patch_size)} patch dimensions for {n_spatial_dims} "
            f"spatial dims)."
            f"Array shape: {arr.shape}, spatial shape: {spatial_shape}, axes: {axes}. "
            f"Check the axes order."
        )

    for i, (patch_dim, img_dim) in enumerate(
        zip(patch_size, spatial_shape, strict=True)
    ):
        if patch_dim > img_dim:
            if n_spatial_dims == 1:
                dim_names = ["X"]
            elif n_spatial_dims == 2:
                dim_names = ["Y", "X"]
            elif n_spatial_dims == 3:
                dim_names = ["Z", "Y", "X"]
            else:
                dim_names = [f"dim_{j}" for j in range(n_spatial_dims)]

            dim_name = dim_names[i] if i < len(dim_names) else f"dim_{i}"

            raise ValueError(
                f"{dim_name} patch size is inconsistent with image shape "
                f"(got {patch_dim} patch size for {img_dim} image dimension). "
                f"Spatial shape: {spatial_shape}, axes: {axes}. Check the axes order."
            )

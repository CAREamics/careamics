"""N2V manipulation functions for PyTorch."""

import torch

from .struct_mask_parameters import StructMaskParameters


def _apply_struct_mask_torch(
    patch: torch.Tensor,
    coords: torch.Tensor,
    struct_params: StructMaskParameters,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply structN2V masks to patch.

    Each point in `coords` corresponds to the center of a mask. Masks are parameterized
    by `struct_params`, and pixels in the mask (with respect to `coords`) are replaced
    by a random value.

    Note that the structN2V mask is applied in 2D at the coordinates given by `coords`.

    Parameters
    ----------
    patch : torch.Tensor
        Patch to be manipulated, (batch, y, x) or (batch, z, y, x).
    coords : torch.Tensor
        Coordinates of the ROI (subpatch) centers.
    struct_params : StructMaskParameters
        Parameters for the structN2V mask (axis and span).
    rng : torch.Generator, optional
        Random number generator.

    Returns
    -------
    torch.Tensor
        Patch with the structN2V mask applied.
    """
    # Handle empty coordinates
    if len(coords) == 0:
        return patch

    if rng is None:
        rng = torch.Generator(device=patch.device)

    # Relative axis
    moving_axis = -1 - struct_params.axis

    # Create a mask array
    mask_shape = [1] * len(patch.shape)
    mask_shape[moving_axis] = struct_params.span
    mask = torch.ones(mask_shape, device=patch.device)

    center = torch.tensor(mask.shape, device=patch.device) // 2

    # Mark the center
    mask[tuple(center)] = 0

    # Displacements from center
    displacements = torch.stack(torch.where(mask == 1)) - center.unsqueeze(1)

    # Combine all coords (ndim, npts) with all displacements (ncoords, ndim)
    mix = displacements.T.unsqueeze(-1) + coords.T.unsqueeze(0)
    mix = mix.permute([1, 0, 2]).reshape([mask.ndim, -1]).T

    # Filter out invalid indices
    valid_indices = (mix[:, moving_axis] >= 0) & (
        mix[:, moving_axis] < patch.shape[moving_axis]
    )
    mix = mix[valid_indices]

    mins = patch.min(-1)[0].min(-1)[0]
    maxs = patch.max(-1)[0].max(-1)[0]

    # Iterate over batch dimension
    # Note: If input was 1D (L,), it should be reshaped to (1, L) before calling this
    # to avoid iterating L times.
    for i in range(patch.shape[0]):
        batch_coords = mix[mix[:, 0] == i]
        if len(batch_coords) == 0:
            continue

        min_ = mins[i].item()
        max_ = maxs[i].item()
        random_values = torch.empty(len(batch_coords), device=patch.device).uniform_(
            min_, max_, generator=rng
        )
        patch[tuple(batch_coords[:, j] for j in range(patch.ndim))] = random_values

    return patch


def _get_stratified_coords_torch(
    mask_pixel_perc: float,
    shape: tuple[int, ...],
    rng: torch.Generator,
) -> torch.Tensor:
    """
    Generate coordinates of the pixels to mask.

    Randomly selects the coordinates of the pixels to mask in a stratified way, i.e.
    the distance between masked pixels is approximately the same. This is achieved by
    defining a grid and sampling a pixel in each grid square. The grid is defined such
    that the resulting density of masked pixels is the desired masked pixel percentage.

    Parameters
    ----------
    mask_pixel_perc : float
        Expected value for percentage of masked pixels across the whole image.
    shape : tuple[int, ...]
        Shape of the input patch including batch dimension.
        (B, Y, X) or (B, Z, Y, X) or (B, L).
    rng : torch.Generator or None
        Random number generator.

    Returns
    -------
    torch.Tensor
        Array of coordinates of the masked pixels with shape (n_points, n_dims).
        The first column is the batch index.
    """
    # Handle edge case where mask percentage is 0
    if mask_pixel_perc <= 0:
        return torch.empty((0, len(shape)), dtype=torch.int32, device=rng.device)

    # Assume shape is always (Batch, Spatial...) due to normalization in caller
    batch_size = shape[0]
    spatial_shape = shape[1:]
    n_dims = len(spatial_shape)

    expected_area_per_pixel = 1 / (mask_pixel_perc / 100)

    # keep the grid size in floats for a more accurate expected masked pixel percentage
    grid_size = expected_area_per_pixel ** (1 / n_dims)
    grid_dims = torch.ceil(torch.tensor(spatial_shape) / grid_size).int()

    # coords on a fixed grid (top left corner)
    # This meshgrid includes the Batch dimension as the first dimension
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(batch_size, dtype=torch.float),
            *[
                torch.arange(0, grid_dims[i].item(), dtype=torch.float) * grid_size
                for i in range(n_dims)
            ],
            indexing="ij",
        ),
        -1,
    ).reshape(-1, n_dims + 1)

    # add random offset to get a random coord in each grid box
    # also keep the offset in floats
    offset = (
        torch.rand((len(coords), n_dims), device=rng.device, generator=rng) * grid_size
    )
    coords = coords.to(rng.device)

    # Add offset to spatial dimensions only (indices 1 to end)
    coords[:, 1:] += offset
    coords = torch.floor(coords).int()

    # filter pixels out of bounds
    # Check spatial dimensions against spatial shape
    out_of_bounds = (
        coords[:, 1:]
        >= torch.tensor(spatial_shape, device=rng.device).reshape(1, n_dims)
    ).any(1)
    coords = coords[~out_of_bounds]

    return coords


def uniform_manipulate_torch(
    patch: torch.Tensor,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    remove_center: bool = True,
    struct_params: StructMaskParameters | None = None,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manipulate pixels by replacing them with a neighbor values.

    Manipulated pixels are selected uniformly in a subpatch, away from a grid
    with an approximate uniform probability to be selected across the whole patch.
    If `struct_params` is not None, an additional structN2V mask is applied to the
    data, replacing the pixels in the mask with random values (excluding the pixel
    already manipulated).

    Supports 1D, 2D, and 3D data.

    Parameters
    ----------
    patch : torch.Tensor
        Image patch, 1D, 2D or 3D, shape (L,), (B, Y, X), or (B, Z, Y, X).
    mask_pixel_percentage : float
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    remove_center : bool
        Whether to remove the center pixel from the subpatch, by default True.
    struct_params : StructMaskParameters or None
        Parameters for the structN2V mask (axis and span).
    rng : torch.Generator or None
        Random number generator.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the manipulated patch and the corresponding mask.
    """
    if rng is None:
        rng = torch.Generator(device=patch.device)

    # 1. Normalize Input: Enforce (Batch, Spatial...) structure
    # If input is 1D unbatched (L,), unsqueeze to (1, L)
    is_unbatched_1d = len(patch.shape) == 1
    if is_unbatched_1d:
        patch = patch.unsqueeze(0)

    transformed_patch = patch.clone()

    # 2. Get Coordinates
    # _get_stratified_coords_torch now always returns (N, 1+SpatialDims)
    subpatch_centers = _get_stratified_coords_torch(
        mask_pixel_percentage, patch.shape, rng
    ).to(device=patch.device)

    # If no pixels to mask, return early
    if len(subpatch_centers) == 0:
        mask = torch.zeros_like(transformed_patch, dtype=torch.uint8)
        if is_unbatched_1d:
            return transformed_patch.squeeze(0), mask.squeeze(0)
        return transformed_patch, mask

    roi_span_full = torch.arange(
        -(subpatch_size // 2),
        subpatch_size // 2 + 1,
        dtype=torch.int32,
        device=patch.device,
    )
    roi_span = roi_span_full[roi_span_full != 0] if remove_center else roi_span_full

    # 3. Calculate Random Increments
    # We only increment spatial dimensions (dim 1 onwards)
    n_spatial_dims = patch.ndim - 1

    random_increment = roi_span[
        torch.randint(
            low=0,
            high=len(roi_span),
            size=(subpatch_centers.shape[0], n_spatial_dims),
            generator=rng,
            device=patch.device,
        )
    ]

    replacement_coords = subpatch_centers.clone()

    # Calculate limits for spatial dims
    spatial_limits = torch.tensor(
        [v - 1 for v in patch.shape[1:]], dtype=torch.int32, device=patch.device
    )

    # Add increments and clamp
    replacement_coords[:, 1:] = torch.clamp(
        replacement_coords[:, 1:] + random_increment,
        min=torch.zeros(n_spatial_dims, dtype=torch.int32, device=patch.device),
        max=spatial_limits,
    )

    # 4. Apply Replacement
    replacement_pixels = patch[tuple(replacement_coords.T)]
    transformed_patch[tuple(subpatch_centers.T)] = replacement_pixels

    mask = (transformed_patch != patch).to(dtype=torch.uint8)

    # 5. Apply Struct Mask
    if struct_params is not None:
        transformed_patch = _apply_struct_mask_torch(
            transformed_patch, subpatch_centers, struct_params, rng
        )

    # 6. Restore Input Shape
    if is_unbatched_1d:
        return transformed_patch.squeeze(0), mask.squeeze(0)

    return transformed_patch, mask


def median_manipulate_torch(
    batch: torch.Tensor,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    struct_params: StructMaskParameters | None = None,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manipulate pixels by replacing them with the median of their surrounding subpatch.

    N2V2 version, manipulated pixels are selected randomly away from a grid with an
    approximate uniform probability to be selected across the whole patch.

    Supports 1D, 2D, and 3D data.

    Parameters
    ----------
    batch : torch.Tensor
        Patch data, 1D, 2D or 3D, shape (L,), (B, Y, X), or (B, Z, Y, X).
    mask_pixel_percentage : float
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    struct_params : StructMaskParameters | None, optional
        Parameters for the structN2V mask (axis and span).
    rng : torch.default_generator or None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the manipulated patch and the mask.
    """
    if rng is None:
        rng = torch.Generator(device=batch.device)

    # 1. Normalize Input: Enforce (Batch, Spatial...) structure
    is_unbatched_1d = len(batch.shape) == 1
    if is_unbatched_1d:
        batch = batch.unsqueeze(0)

    # 2. Get Coordinates
    subpatch_center_coordinates = _get_stratified_coords_torch(
        mask_pixel_percentage, batch.shape, rng
    ).to(device=batch.device)

    if len(subpatch_center_coordinates) == 0:
        mask = torch.zeros_like(batch, dtype=torch.uint8)
        output_batch = batch.clone()
        if is_unbatched_1d:
            return output_batch.squeeze(0), mask.squeeze(0)
        return output_batch, mask

    # 3. Create Offsets for ROIs
    pad_value = subpatch_size // 2
    n_spatial_dims = batch.ndim - 1

    offset_tensors = [
        torch.arange(-pad_value, pad_value + 1, device=batch.device)
        for _ in range(n_spatial_dims)
    ]

    if n_spatial_dims == 1:
        # Special case for 1D spatial dim
        offsets = offset_tensors[0].unsqueeze(1)  # (subpatch_size, 1)
    else:
        offsets = torch.stack(
            torch.meshgrid(*offset_tensors, indexing="ij"), dim=-1
        ).reshape(-1, n_spatial_dims)

    # 4. Gather ROIs (Regions of Interest)
    coords_axes = []
    coords_expands = []

    # Iterate over all dimensions (0 is batch, 1..N are spatial)
    for d in range(batch.ndim):
        coords_axes.append(subpatch_center_coordinates[:, d])
        if d == 0:
            # Batch dimension: expand without numerical offset
            coords_expands.append(
                subpatch_center_coordinates[:, d]
                .unsqueeze(1)
                .expand(-1, offsets.shape[0])
            )
        else:
            # Spatial dimensions: add offsets
            # offsets column index is d-1 because offsets tensor has no batch dim
            coords_expands.append(
                (
                    subpatch_center_coordinates[:, d].unsqueeze(1) + offsets[:, d - 1]
                ).clamp(0, batch.shape[d] - 1)
            )

    rois = batch[tuple(coords_expands)]

    # 5. Filter ROIs (Struct or Center Pixel)
    if struct_params is not None:
        # Create grid for the subpatch to determine which pixels to mask out
        spatial_grids = torch.meshgrid(
            *[
                torch.arange(subpatch_size, device=batch.device)
                for _ in range(n_spatial_dims)
            ],
            indexing="ij",
        )

        center_idx = subpatch_size // 2
        halfspan = (struct_params.span - 1) // 2

        if struct_params.axis >= n_spatial_dims:
            raise ValueError(
                f"Struct axis {struct_params.axis} out of bounds for {n_spatial_dims}D spatial data"
            )

        # Logic to mask out the 'span' along the axis
        # We want to keep pixels that are NOT in the struct mask line
        mask_condition = torch.ones(
            spatial_grids[0].shape, dtype=torch.bool, device=batch.device
        )

        for i in range(n_spatial_dims):
            if i == struct_params.axis:
                # The masked axis must be within the span
                mask_condition &= (spatial_grids[i] >= center_idx - halfspan) & (
                    spatial_grids[i] <= center_idx + halfspan
                )
            else:
                # Other axes must be exactly at the center
                mask_condition &= spatial_grids[i] == center_idx

        keep_mask = (~mask_condition).flatten()
        rois_filtered = rois[:, keep_mask]
    else:
        # Simple N2V: Remove center pixel
        center_idx = (subpatch_size**n_spatial_dims) // 2
        rois_filtered = torch.cat(
            [rois[:, :center_idx], rois[:, center_idx + 1 :]], dim=1
        )

    # 6. Compute Median and Update
    medians = rois_filtered.median(dim=1).values

    output_batch = batch.clone()
    output_batch[tuple(coords_axes)] = medians
    mask = torch.where(output_batch != batch, 1, 0).to(torch.uint8)

    # 7. Apply Struct Mask (if active)
    if struct_params is not None:
        output_batch = _apply_struct_mask_torch(
            output_batch, subpatch_center_coordinates, struct_params, rng
        )

    # 8. Restore Input Shape
    if is_unbatched_1d:
        return output_batch.squeeze(0), mask.squeeze(0)

    return output_batch, mask

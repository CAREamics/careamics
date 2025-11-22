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
    for i in range(patch.shape[0]):
        batch_coords = mix[mix[:, 0] == i]
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

    Supports 1D, 2D, and 3D data. For 1D data (shape: (L,)), treats it as having
    a single batch dimension to maintain consistent coordinate format.

    Parameters
    ----------
    mask_pixel_perc : float
        Expected value for percentage of masked pixels across the whole image.
    shape : tuple[int, ...]
        Shape of the input patch. Can be 1D (L,), 2D (Y, X), or 3D (Z, Y, X).
    rng : torch.Generator or None
        Random number generator.

    Returns
    -------
    torch.Tensor
        Array of coordinates of the masked pixels with shape (n_points, n_dims).
        For 1D input, n_dims=1; for 2D input, n_dims=2; for 3D input, n_dims=3.
    """
    # Handle edge case where mask percentage is 0
    if mask_pixel_perc <= 0:
        return torch.empty((0, len(shape)), dtype=torch.int32, device=rng.device)

    # Implementation logic:
    #    find a box size s.t sampling 1 pixel within the box will result in the desired
    # pixel percentage. Make a grid of these boxes that cover the patch (the area of
    # the grid will be greater than or equal to the area of the patch) and sample 1
    # pixel in each box. The density of masked pixels is an intensive property therefore
    # any subset of this area will have the desired expected masked pixel percentage.
    # We can get our desired patch with our desired expected masked pixel percentage by
    # simply filtering out masked pixels that lie outside of our patch bounds.

    # For 1D data, shape is (L,) so we treat all dimensions as spatial
    # For 2D+ data, first dimension is treated as batch-like
    if len(shape) == 1:
        # 1D case: no batch dimension, just signal length
        batch_size = 1  # Dummy batch for consistent handling
        spatial_shape = shape  # All dimensions are spatial
    else:
        # 2D/3D case: first dimension treated as batch
        batch_size = shape[0]
        spatial_shape = shape[1:]

    n_dims = len(spatial_shape)
    expected_area_per_pixel = 1 / (mask_pixel_perc / 100)

    # keep the grid size in floats for a more accurate expected masked pixel percentage
    grid_size = expected_area_per_pixel ** (1 / n_dims)
    grid_dims = torch.ceil(torch.tensor(spatial_shape) / grid_size).int()

    # coords on a fixed grid (top left corner)
    if len(shape) == 1:
        # 1D case: generate coordinates without batch dimension
        grid_tensors = [
            torch.arange(0, grid_dims[i].item(), dtype=torch.float) * grid_size
            for i in range(n_dims)
        ]
        if len(grid_tensors) == 1:
            # Special handling for single dimension
            coords = grid_tensors[0].unsqueeze(-1)
        else:
            coords = torch.stack(
                torch.meshgrid(*grid_tensors, indexing="ij"), -1
            ).reshape(-1, n_dims)
    else:
        # 2D/3D case: include batch dimension in coordinates
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
        torch.rand((len(coords), n_dims), device=rng.device, generator=rng)
        * grid_size
    )
    coords = coords.to(rng.device)

    if len(shape) == 1:
        # 1D case: add offset to all dimensions (no batch dimension)
        coords += offset
    else:
        # 2D/3D case: add offset only to spatial dimensions, not batch
        coords[:, 1:] += offset

    coords = torch.floor(coords).int()

    # filter pixels out of bounds
    if len(shape) == 1:
        # 1D case: check all dimensions
        out_of_bounds = (
            coords >= torch.tensor(spatial_shape, device=rng.device).reshape(1, n_dims)
        ).any(1)
    else:
        # 2D/3D case: check only spatial dimensions (skip batch dimension)
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
        Image patch, 1D, 2D or 3D, shape (L,), (Y, X), or (Z, Y, X).
    mask_pixel_percentage : float
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    remove_center : bool
        Whether to remove the center pixel from the subpatch, by default False.
    struct_params : StructMaskParameters or None
        Parameters for the structN2V mask (axis and span).
    rng : torch.default_generator or None
        Random number generator.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        tuple containing the manipulated patch and the corresponding mask.
    """
    if rng is None:
        rng = torch.Generator(device=patch.device)
        # TODO do we need seed ?

    # create a copy of the patch
    transformed_patch = patch.clone()

    # get the coordinates of the pixels to be masked
    subpatch_centers = _get_stratified_coords_torch(
        mask_pixel_percentage, patch.shape, rng
    )
    subpatch_centers = subpatch_centers.to(device=patch.device)

    # If no pixels to mask, return original patch and empty mask
    if len(subpatch_centers) == 0:
        mask = torch.zeros_like(patch, dtype=torch.uint8)
        return transformed_patch, mask

    # TODO refactor with non negative indices?
    # arrange the list of indices to represent the ROI around the pixel to be masked
    roi_span_full = torch.arange(
        -(subpatch_size // 2),
        subpatch_size // 2 + 1,
        dtype=torch.int32,
        device=patch.device,
    )

    # remove the center pixel from the ROI
    roi_span = roi_span_full[roi_span_full != 0] if remove_center else roi_span_full

    # Determine dimensionality for proper coordinate handling
    is_1d = len(patch.shape) == 1
    n_coord_dims = subpatch_centers.shape[1]

    # create a random increment to select the replacement value
    # this increment is added to the center coordinates
    random_increment = roi_span[
        torch.randint(
            low=min(roi_span),
            high=max(roi_span) + 1,
            # For 1D: all dimensions; For 2D/3D: skip first (batch) dimension
            size=(subpatch_centers.shape[0], n_coord_dims),
            generator=rng,
            device=patch.device,
        )
    ]

    # compute the replacement pixel coordinates
    replacement_coords = subpatch_centers.clone()

    if is_1d:
        # 1D case: add increment to all dimensions
        replacement_coords = torch.clamp(
            replacement_coords + random_increment,
            torch.zeros(len(patch.shape), dtype=torch.int32, device=patch.device),
            torch.tensor(
                [v - 1 for v in patch.shape], dtype=torch.int32, device=patch.device
            ),
        )
    else:
        # 2D/3D case: add increment only to spatial dimensions, not batch
        replacement_coords[:, 1:] = torch.clamp(
            replacement_coords[:, 1:] + random_increment,
            torch.zeros(
                len(patch.shape[1:]), dtype=torch.int32, device=patch.device
            ),
            torch.tensor(
                [v - 1 for v in patch.shape[1:]], dtype=torch.int32,
                device=patch.device
            ),
        )

    # replace the pixels in the patch
    # tuples and transpose are needed for proper indexing
    replacement_pixels = patch[tuple(replacement_coords.T)]
    transformed_patch[tuple(subpatch_centers.T)] = replacement_pixels

    # create a mask representing the masked pixels
    mask = (transformed_patch != patch).to(dtype=torch.uint8)

    # apply structN2V mask if needed
    if struct_params is not None:
        transformed_patch = _apply_struct_mask_torch(
            transformed_patch, subpatch_centers, struct_params, rng
        )

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

    If `struct_params` is not None, an additional structN2V mask is applied to the data,
    replacing the pixels in the mask with random values (excluding the pixel already
    manipulated).

    Supports 1D, 2D, and 3D data.

    Parameters
    ----------
    batch : torch.Tensor
        Patch data, 1D, 2D or 3D, shape (L,), (Y, X), or (Z, Y, X).
    mask_pixel_percentage : float
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    struct_params : StructMaskParameters or None, optional
        Parameters for the structN2V mask (axis and span).
    rng : torch.default_generator or None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        tuple containing the manipulated patch and the mask.
    """
    if rng is None:
        rng = torch.Generator(device=batch.device)

    # get the coordinates of the future ROI centers
    subpatch_center_coordinates = _get_stratified_coords_torch(
        mask_pixel_percentage, batch.shape, rng
    ).to(
        device=batch.device
    )  # (num_coordinates, batch + num_spatial_dims)

    # If no pixels to mask, return original batch and empty mask
    if len(subpatch_center_coordinates) == 0:
        mask = torch.zeros_like(batch, dtype=torch.uint8)
        return batch, mask

    # Calculate the padding value for the input tensor
    pad_value = subpatch_size // 2

    # Determine if this is 1D data
    is_1d = len(batch.shape) == 1
    n_coord_dims = subpatch_center_coordinates.shape[1]

    # Generate all offsets for the ROIs
    if is_1d:
        # 1D case: all dimensions are spatial, no batch dimension to skip
        offset_tensor = torch.arange(
            -pad_value, pad_value + 1, device=batch.device
        )
        offsets = offset_tensor.unsqueeze(-1)  # (subpatch_size, 1)
    else:
        # 2D/3D case: skip first dimension (batch), generate offsets for spatial dims
        offset_tensors = [
            torch.arange(-pad_value, pad_value + 1, device=batch.device)
            for _ in range(1, n_coord_dims)
        ]
        offsets = torch.stack(
            torch.meshgrid(*offset_tensors, indexing="ij"), dim=-1
        )
        offsets = offsets.reshape(-1, len(offset_tensors))

    # Create the list to assemble coordinates of the ROIs centers for each axis
    coords_axes = []
    # Create the list to assemble the span of coordinates defining the ROIs for each
    # axis
    coords_expands = []

    if is_1d:
        # 1D case: simple offset application
        for d in range(n_coord_dims):
            coords_axes.append(subpatch_center_coordinates[:, d])
            # Expand coordinates with offsets
            coords_expands.append(
                (
                    subpatch_center_coordinates[:, d].unsqueeze(1)
                    + offsets[:, d]
                ).clamp(0, batch.shape[d] - 1)
            )
    else:
        # 2D/3D case: treat first dimension as batch
        for d in range(n_coord_dims):
            coords_axes.append(subpatch_center_coordinates[:, d])
            if d == 0:
                # For batch dimension coordinates are not expanded (no offsets)
                coords_expands.append(
                    subpatch_center_coordinates[:, d]
                    .unsqueeze(1)
                    .expand(-1, offsets.shape[0])
                )
            else:
                # For spatial dimensions, coordinates are expanded with offsets
                coords_expands.append(
                    (
                        subpatch_center_coordinates[:, d].unsqueeze(1)
                        + offsets[:, d - 1]
                    ).clamp(0, batch.shape[d] - 1)
                )

    # create array of rois by indexing the batch with gathered coordinates
    rois = batch[
        tuple(coords_expands)
    ]  # (num_coordinates, subpatch_size**num_spacial_dims)

    if struct_params is not None:
        if is_1d:
            # For 1D, struct_params doesn't make sense, so just remove center
            center_idx = subpatch_size // 2
            rois_filtered = torch.cat(
                [rois[:, :center_idx], rois[:, center_idx + 1 :]], dim=1
            )
        else:
            # Create the structN2V mask for 2D/3D
            h, w = torch.meshgrid(
                torch.arange(subpatch_size),
                torch.arange(subpatch_size),
                indexing="ij",
            )
            center_idx = subpatch_size // 2
            halfspan = (struct_params.span - 1) // 2

            # Determine the axis along which to apply the mask
            if struct_params.axis == 0:
                center_axis = h
                span_axis = w
            else:
                center_axis = w
                span_axis = h

            # Create the mask
            struct_mask = (
                ~(
                    (center_axis == center_idx)
                    & (span_axis >= center_idx - halfspan)
                    & (span_axis <= center_idx + halfspan)
                )
            ).flatten()
            rois_filtered = rois[:, struct_mask]
    else:
        # Remove the center pixel value from the rois
        center_idx = subpatch_size // 2
        rois_filtered = torch.cat(
            [rois[:, :center_idx], rois[:, center_idx + 1 :]], dim=1
        )

    # compute the medians.
    medians = rois_filtered.median(dim=1).values  # (num_coordinates,)

    # Update the output tensor with medians
    output_batch = batch.clone()
    output_batch[tuple(coords_axes)] = medians
    mask = torch.where(output_batch != batch, 1, 0).to(torch.uint8)

    if struct_params is not None:
        output_batch = _apply_struct_mask_torch(
            output_batch, subpatch_center_coordinates, struct_params
        )

    return output_batch, mask

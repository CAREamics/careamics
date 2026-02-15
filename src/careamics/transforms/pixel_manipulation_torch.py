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

    # flatten in spatial dims to find min and max for each patch in the batch
    batch_mins = patch.view(patch.shape[0], -1).min(dim=-1).values
    batch_maxs = patch.view(patch.shape[0], -1).max(dim=-1).values
    for i in range(patch.shape[0]):
        batch_coords = mix[mix[:, 0] == i]
        min_ = batch_mins[i].item()
        max_ = batch_maxs[i].item()
        random_values = torch.empty(len(batch_coords), device=patch.device).uniform_(
            min_, max_, generator=rng
        )
        patch[tuple(batch_coords[:, i] for i in range(patch.ndim))] = random_values

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
        Shape of the input patch.
    rng : torch.Generator or None
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of coordinates of the masked pixels.
    """
    # Implementation logic:
    #    find a box size s.t sampling 1 pixel within the box will result in the desired
    # pixel percentage. Make a grid of these boxes that cover the patch (the area of
    # the grid will be greater than or equal to the area of the patch) and sample 1
    # pixel in each box. The density of masked pixels is an intensive property therefore
    # any subset of this area will have the desired expected masked pixel percentage.
    # We can get our desired patch with our desired expected masked pixel percentage by
    # simply filtering out masked pixels that lie outside of our patch bounds.

    batch_size = shape[0]
    spatial_shape = shape[1:]

    n_dims = len(spatial_shape)
    expected_area_per_pixel = 1 / (mask_pixel_perc / 100)

    # keep the grid size in floats for a more accurate expected masked pixel percentage
    grid_size = expected_area_per_pixel ** (1 / n_dims)
    grid_dims = torch.ceil(torch.tensor(spatial_shape) / grid_size).int()

    # coords on a fixed grid (top left corner)
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(batch_size, dtype=torch.float),
            *[torch.arange(0, grid_dims[i].item()) * grid_size for i in range(n_dims)],
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
    coords[:, 1:] += offset
    coords = torch.floor(coords).int()

    # filter pixels out of bounds
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

    # TODO add more details, especially about batch

    Manipulated pixels are selected uniformly selected in a subpatch, away from a grid
    with an approximate uniform probability to be selected across the whole patch.
    If `struct_params` is not None, an additional structN2V mask is applied to the
    data, replacing the pixels in the mask with random values (excluding the pixel
    already manipulated).

    Parameters
    ----------
    patch : torch.Tensor
        Image patch, 2D or 3D, shape (y, x) or (z, y, x). # TODO batch and channel.
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

    # create a random increment to select the replacement value
    # this increment is added to the center coordinates
    random_increment = roi_span[
        torch.randint(
            low=min(roi_span),
            high=max(roi_span) + 1,
            # one less coord dim: we shouldn't add a random increment to the batch coord
            size=(subpatch_centers.shape[0], subpatch_centers.shape[1] - 1),
            generator=rng,
            device=patch.device,
        )
    ]

    # compute the replacement pixel coordinates
    replacement_coords = subpatch_centers.clone()
    # only add random increment to the spatial dimensions, not the batch dimension
    replacement_coords[:, 1:] = torch.clamp(
        replacement_coords[:, 1:] + random_increment,
        torch.zeros_like(torch.tensor(patch.shape[1:])).to(device=patch.device),
        torch.tensor([v - 1 for v in patch.shape[1:]]).to(device=patch.device),
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

    Parameters
    ----------
    batch : torch.Tensor
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
           tuple containing the manipulated patch, the original patch and the mask.
    """
    # -- Implementation summary
    # 1. Generate coordinates that correspond to the pixels chosen for masking.
    # 2. Subpatches are extracted, where the coordinate to mask is at the center.
    # 3. The medians of these subpatches are calculated, but we do not want to include
    #    the original pixel in the calculation so we mask it. In the case of StructN2V,
    #    we do not include any pixels in the struct mask in the median calculation.

    if rng is None:
        rng = torch.Generator(device=batch.device)

    # resulting center coord shape: (num_coordinates, batch + num_spatial_dims)
    subpatch_center_coordinates = _get_stratified_coords_torch(
        mask_pixel_percentage, batch.shape, rng
    )
    # pixel coordinates of all the subpatches
    # shape: (num_coordinates, subpatch_size, subpatch_size, ...)
    subpatch_coords = _get_subpatch_coords(
        subpatch_center_coordinates, subpatch_size, batch.shape
    )

    # this indexes and stacks all the subpatches along the first dimension
    # subpatches shape: (num_coordinates, subpatch_size, subpatch_size, ...)
    subpatches = batch[tuple(subpatch_coords)]

    ndims = batch.ndim - 1
    # subpatch mask to exclude values from median calculation
    if struct_params is None:
        subpatch_mask = _create_center_pixel_mask(ndims, subpatch_size, batch.device)
    else:
        subpatch_mask = _create_struct_mask(
            ndims, subpatch_size, struct_params, batch.device
        )
    subpatches_masked = subpatches[:, subpatch_mask]

    medians = subpatches_masked.median(dim=1).values  # (num_coordinates,)

    # Update the output tensor with medians
    output_batch = batch.clone()
    output_batch[tuple(subpatch_center_coordinates.T)] = medians
    mask = (batch != output_batch).to(torch.uint8)

    if struct_params is not None:
        output_batch = _apply_struct_mask_torch(
            output_batch, subpatch_center_coordinates, struct_params
        )

    return output_batch, mask


def _create_center_pixel_mask(
    ndims: int, subpatch_size: int, device: torch.device
) -> torch.Tensor:
    """
    Create a mask for the center pixel of a subpatch.

    Parameters
    ----------
    ndims : int
        The number of dimensions.
    subpatch_size : int
        The size of one dimension of the subpatch. The created mask must be the same
        size as the subpatch. Cannot be an even number.
    device : torch.device
        Device to create the mask on, e.g. "cuda".

    Returns
    -------
    torch.Tensor
        Tensor of bools. False where pixels should be masked and True otherwise.
    """
    if subpatch_size % 2 == 0:
        raise ValueError("`subpatch` size cannot be even.")
    subpatch_shape = (subpatch_size,) * ndims
    centre_idx = (subpatch_size // 2,) * ndims
    cp_mask = torch.ones(subpatch_shape, dtype=torch.bool, device=device)
    cp_mask[centre_idx] = False
    return cp_mask


def _create_struct_mask(
    ndims: int,
    subpatch_size: int,
    struct_params: StructMaskParameters,
    device: torch.device,
) -> torch.Tensor:
    """
    Create the mask for StructN2V.

    Parameters
    ----------
    ndims : int
        The number of dimensions.
    subpatch_size : int
        The size of one dimension of the subpatch. The created mask must be the same
        size as the subpatch. Cannot be an even number.
    struct_params : StructMaskParameters
        Parameters for the structN2V mask (axis and span).
    device : torch.device
        Device to create the mask on, e.g. "cuda".

    Returns
    -------
    torch.Tensor
        Tensor of bools. False where pixels should be masked and True otherwise.
    """
    if subpatch_size % 2 == 0:
        raise ValueError("`subpatch` size cannot be even.")
    center_idx = subpatch_size // 2
    span_start = (subpatch_size - struct_params.span) // 2
    span_end = subpatch_size - span_start  # symmetric
    span_axis = ndims - 1 - struct_params.axis  # e.g. horizontal is the last axis

    struct_mask = torch.ones((subpatch_size,) * ndims, dtype=torch.bool, device=device)
    # indexes the center unless it is the axis on which the struct mask spans
    struct_slice = (
        center_idx if d != span_axis else slice(span_start, span_end)
        for d in range(ndims)
    )
    struct_mask[*struct_slice] = False
    return struct_mask


def _get_subpatch_coords(
    subpatch_centers: torch.Tensor, subpatch_size: int, batch_shape: tuple[int, ...]
) -> torch.Tensor:
    """Get pixel coordinates for subpatches with centers at `subpatch_centers`.

    The coordinates are returned in the shape `(D ,N, S, S)` or `(D, N, S, S, S)` for
    2D and 3D patches respectively, where `D` is the number of dimension including the
    batch dimension, `N` is the number of subpatches, and `S` is the
    subpatch size. N is determined from the length of `subpatch_centres`.

    If a subpatch would overlap the bounds of the patch, the coordinates are clipped.
    This does result in some duplicated coordinates on the boundary of the patch.

    Parameters
    ----------
    subpatch_centers : torch.Tensor
        Coordinates of the center of a subpatch, including the batch dimension, i.e.
        (b, (z), y, x). Has shape (N, D) for N different subpatch centers and D
        dimensions.
    subpatch_size : int
        The size of one dimension of the subpatch. The created mask must be the same
        size as the subpatch.
    batch_shape : tuple[int, ...]
        The shape of the batch that is being processed, i.e. (B ,(Z), Y, X).

    Returns
    -------
    torch.Tensor
        The coordinates of every pixel in each subpatch, stacked into the shape
        `(D ,N, S, S)` or `(D, N, S, S, S)` for 2D and 3D patches respectively, where
        `D` is the number of dimension including the batch dimension, `N` is the number
        of subpatches, and `S` is the subpatch size.
    """
    device = subpatch_centers.device
    ndims = len(batch_shape) - 1  # spatial dimensions

    half_size = subpatch_size // 2
    # pixel offset from the center of the subpatch, i.e. coords relative to the center
    offsets = torch.meshgrid(
        [torch.arange(-half_size, half_size + 1, device=device) for _ in range(ndims)],
        indexing="ij",
    )
    # add zero offset for the batch dimension
    subpatch_shape = (subpatch_size,) * ndims
    offsets = torch.stack(
        [torch.zeros(subpatch_shape, dtype=torch.int64, device=device), *offsets], dim=0
    )

    # now we need to add the offset to the subpatch_centers to get the subpatch coords
    # subpatch_shape: (n_centres, ndims + 1)
    # offset_shape: (ndims + 1, subpatch_size, subpatch_size, ...)
    # we need to add singleton dims to broadcast the tensors
    subpatch_centers = subpatch_centers[..., *(torch.newaxis for _ in range(ndims))]
    offsets = offsets[torch.newaxis]

    # resulting shape: (n_centres, ndims + 1, subpatch_size, subpatch_size, ...)
    subpatch_coords = subpatch_centers + offsets
    subpatch_coords = torch.swapaxes(subpatch_coords, 0, 1)
    # clamp coordinates so they are not outside the bounds of the patch
    broadcast_shape = (ndims + 1, *(1 for _ in range(ndims + 1)))
    minimum = torch.zeros(broadcast_shape, dtype=torch.int64, device=device)
    maximum = torch.tensor(batch_shape, device=device).reshape(broadcast_shape) - 1
    subpatch_coords = subpatch_coords.clamp(minimum, maximum)
    return subpatch_coords

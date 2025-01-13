from typing import Optional

import torch

from .struct_mask_parameters import StructMaskParameters


def _apply_struct_mask_torch(
    patch: torch.Tensor,
    coords: torch.Tensor,
    struct_params: StructMaskParameters,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply structN2V masks to patch.

    Each point in `coords` corresponds to the center of a mask. Masks are parameterized
    by `struct_params`, and pixels in the mask (with respect to `coords`) are replaced
    by a random value.

    Note that the structN2V mask is applied in 2D at the coordinates given by `coords`.

    Parameters
    ----------
    patch : torch.Tensor
        Patch to be manipulated, 2D or 3D.
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
        rng = torch.default_generator

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

    # Replace neighboring pixels with random values from a uniform distribution
    random_values = torch.empty(len(mix), device=patch.device).uniform_(
        patch.min().item(), patch.max().item(), generator=rng
    )
    patch[tuple(mix.T.tolist())] = random_values

    return patch


def _odd_jitter_func_torch(step: float, rng: torch.Generator) -> torch.Tensor:
    step_floor = torch.floor(torch.tensor(step))
    odd_jitter = (
        step_floor
        if step_floor == step
        else torch.randint(high=2, size=(1,), generator=rng)
    )
    return step_floor if odd_jitter == 0 else step_floor + 1


def _get_stratified_coords_torch(
    mask_pixel_perc: float,
    shape: tuple[int, ...],
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if len(shape) < 2 or len(shape) > 3:
        raise ValueError(
            "Calculating coordinates is only possible for 2D and 3D patches"
        )

    if rng is None:
        rng = torch.default_generator

    mask_pixel_distance = round((100 / mask_pixel_perc) ** (1 / len(shape)))

    pixel_coords = []
    steps = []
    for axis_size in shape:
        num_pixels = int(torch.ceil(torch.tensor(axis_size / mask_pixel_distance)))
        axis_pixel_coords = torch.linspace(
            0,
            axis_size - (axis_size // num_pixels),
            num_pixels,
            dtype=torch.int32,
            device="cpu",
        )
        step = axis_pixel_coords[1] - axis_pixel_coords[0]
        pixel_coords.append(axis_pixel_coords)
        steps.append(step)

    coordinate_grid_list = torch.meshgrid(*pixel_coords, indexing="ij")
    coordinate_grid = torch.stack([g.flatten() for g in coordinate_grid_list], dim=-1)

    random_increment = torch.randint(
        high=int(_odd_jitter_func_torch(float(max(steps)), rng)),
        size=torch.tensor(coordinate_grid.shape).tolist(),
        generator=rng,
    )
    coordinate_grid += random_increment

    return torch.clamp(
        coordinate_grid,
        torch.zeros_like(torch.tensor(shape)),
        torch.tensor([v - 1 for v in shape]),
    )


def _create_subpatch_center_mask(
    subpatch: torch.Tensor, center_coords: torch.Tensor
) -> torch.Tensor:
    """Create a mask with the center of the subpatch masked.

    Parameters
    ----------
    subpatch : np.ndarray
        Subpatch to be manipulated.
    center_coords : np.ndarray
        Coordinates of the original center before possible crop.

    Returns
    -------
    np.ndarray
        Mask with the center of the subpatch masked.
    """
    mask = torch.ones(torch.tensor(subpatch.shape).tolist())
    mask[tuple(center_coords)] = 0
    return (mask != 0).to(torch.bool)


def _create_subpatch_struct_mask(
    subpatch: torch.Tensor,
    center_coords: torch.Tensor,
    struct_params: StructMaskParameters,
) -> torch.Tensor:
    """Create a structN2V mask for the subpatch.

    Parameters
    ----------
    subpatch : np.ndarray
        Subpatch to be manipulated.
    center_coords : np.ndarray
        Coordinates of the original center before possible crop.
    struct_params : StructMaskParameters
        Parameters for the structN2V mask (axis and span).

    Returns
    -------
    np.ndarray
        StructN2V mask for the subpatch.
    """
    # TODO no test for this function!
    # Create a mask with the center of the subpatch masked
    mask_placeholder = torch.ones(subpatch.shape)

    # reshape to move the struct axis to the first position
    mask_reshaped = torch.permute(mask_placeholder, struct_params.axis, 0)

    # create the mask index for the struct axis
    mask_index = slice(
        max(0, center_coords.take(struct_params.axis) - (struct_params.span - 1) // 2),
        min(
            1 + center_coords.take(struct_params.axis) + (struct_params.span - 1) // 2,
            subpatch.shape[struct_params.axis],
        ),
    )
    mask_reshaped[struct_params.axis][mask_index] = 0

    # reshape back to the original shape
    mask = torch.permute(mask_reshaped, 0, struct_params.axis)

    return (mask != 0).to(torch.bool)  # type: ignore


def uniform_manipulate_torch(
    patch: torch.Tensor,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    remove_center: bool = True,
    struct_params: Optional[StructMaskParameters] = None,
    rng: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manipulate pixels by replacing them with a neighbor values.

    Manipulated pixels are selected unformly selected in a subpatch, away from a grid
    with an approximate uniform probability to be selected across the whole patch.
    If `struct_params` is not None, an additional structN2V mask is applied to the
    data, replacing the pixels in the mask with random values (excluding the pixel
    already manipulated).

    Parameters
    ----------
    patch : torch.Tensor
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
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
        rng = torch.default_generator
        # TODO do we need seed ?

    transformed_patch = patch.clone()
    subpatch_centers = _get_stratified_coords_torch(
        mask_pixel_percentage, patch.shape, rng
    )

    roi_span_full = torch.arange(
        -(subpatch_size // 2),
        subpatch_size // 2 + 1,
        dtype=torch.int32,
        device=patch.device,
    )
    roi_span = roi_span_full[roi_span_full != 0] if remove_center else roi_span_full

    random_increment = roi_span[
        torch.randint(
            low=min(roi_span),
            high=max(roi_span),
            size=subpatch_centers.shape,
            generator=rng,
        )
    ]
    replacement_coords = torch.clamp(
        subpatch_centers + random_increment,
        torch.zeros_like(torch.tensor(patch.shape)),
        torch.tensor([v - 1 for v in patch.shape]),
    )

    replacement_pixels = patch[tuple(replacement_coords.T)]
    transformed_patch[tuple(subpatch_centers.T)] = replacement_pixels

    mask = (transformed_patch != patch).to(dtype=torch.uint8)

    if struct_params is not None:
        transformed_patch = _apply_struct_mask_torch(
            transformed_patch, subpatch_centers, struct_params, rng
        )

    return transformed_patch, mask


def median_manipulate_torch(
    patch: torch.Tensor,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    struct_params: Optional[StructMaskParameters] = None,
    rng: Optional[torch.Generator] = None,
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
    patch : torch.Tensor
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
    mask_pixel_percentage : float
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    struct_params : StructMaskParameters or None, optional
        Parameters for the structN2V mask (axis and span).
    rng : torch.default_generator or None, optional
        Random number generato, by default None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
           tuple containing the manipulated patch, the original patch and the mask.
    """
    if rng is None:
        rng = torch.default_generator

    transformed_patch = patch.clone()
    subpatch_centers = _get_stratified_coords_torch(
        mask_pixel_percentage, patch.shape, rng
    )

    roi_span = torch.tensor(
        [-(subpatch_size // 2), (subpatch_size // 2) + 1], device=patch.device
    )

    subpatch_crops_span_full = subpatch_centers[None, ...].T + roi_span

    subpatch_crops_span_clipped = torch.clamp(
        subpatch_crops_span_full,
        torch.zeros_like(torch.tensor(patch.shape))[:, None, None],
        torch.tensor(patch.shape)[:, None, None],
    )

    for idx in range(subpatch_crops_span_clipped.shape[1]):
        subpatch_coords = subpatch_crops_span_clipped[:, idx, ...]
        idxs = [
            slice(x[0], x[1]) if x[1] - x[0] > 0 else slice(0, 1)
            for x in subpatch_coords
        ]
        subpatch = patch[tuple(idxs)]
        subpatch_center_adjusted = subpatch_centers[idx] - subpatch_coords[:, 0]

        if struct_params is None:
            subpatch_mask = _create_subpatch_center_mask(
                subpatch, subpatch_center_adjusted
            )
        else:
            subpatch_mask = _create_subpatch_struct_mask(
                subpatch, subpatch_center_adjusted, struct_params
            )
        transformed_patch[tuple(subpatch_centers[idx])] = torch.median(
            subpatch[subpatch_mask]
        )

    mask = torch.where(transformed_patch != patch, 1, 0).to(torch.uint8)

    if struct_params is not None:
        transformed_patch = _apply_struct_mask_torch(
            transformed_patch, subpatch_centers, struct_params
        )

    return (
        transformed_patch,
        mask,
    )

    mask = (transformed_patch != patch).to(dtype=torch.uint8)

    return transformed_patch, mask

"""Functions for applying filtering to a stratified patching strategy."""

from tqdm import tqdm

from careamics.config.data.patch_filter.filter_config import FilterConfig
from careamics.config.data.patch_filter.mask_filter_config import MaskFilterConfig
from careamics.utils import get_logger

from .image_stack import ImageStack
from .patch_extractor import PatchExtractor
from .patch_filter import MaskFilter, create_patch_filter
from .patching_strategies import StratifiedPatchingStrategy

logger = get_logger("Patch filtering")


def filter_background(
    patching_strategy: StratifiedPatchingStrategy,
    input_extractor: PatchExtractor[ImageStack],
    patch_filter_config: FilterConfig,
    ref_channel: int,
    bg_relative_prob: float = 0.1,
) -> None:
    """
    Apply filtering to the `patching_strategy` with a `patch_filter`.

    Parameters
    ----------
    patching_strategy : StratifiedPatchingStrategy
        A stratified patching strategy to filter.
    input_extractor : PatchExtractor
        A patch extractor holding the data to filtering.
    patch_filter_config : PatchFilterConfig
        Configuration for filtering patches.
    ref_channel : int
        On which channel to filter the data.
    bg_relative_prob : float
        The probability that a region determined as background will be sampled from each
        epoch.
    """
    patch_filter = create_patch_filter(patch_filter_config)
    patch_size = patching_strategy.patch_size
    region_size = tuple(ps * 2 for ps in patch_size)
    all_grid_coords = patching_strategy.get_all_grid_coords().items()
    n_patches = patching_strategy.n_patches
    n_filtered = 0
    for (data_idx, sample_idx), grid_coords in tqdm(
        all_grid_coords, desc="Filtering background patches with filtering function."
    ):
        probs: dict[tuple[int, ...], float] = {}
        for coords in grid_coords:
            region_coords = tuple(
                ps * c for ps, c in zip(patch_size, coords, strict=True)
            )
            patch = input_extractor.extract_channel_patch(
                data_idx,
                sample_idx=sample_idx,
                channels=[ref_channel],
                coords=region_coords,
                patch_size=region_size,
            )
            if patch_filter.filter_out(patch):
                probs[coords] = bg_relative_prob
                n_filtered += 1
        patching_strategy.set_region_probs(data_idx, sample_idx, probs)
    reduced_patches = patching_strategy.n_patches
    logger.info(
        f"Found {n_filtered} background regions. Number of patches has been reduced to "
        f"{reduced_patches} from {n_patches}."
    )


def filter_background_with_mask(
    patching_strategy: StratifiedPatchingStrategy,
    mask_filter_config: MaskFilterConfig,
    mask_extractor: PatchExtractor[ImageStack],
    bg_relative_prob: float = 0.1,
) -> None:
    """
    Apply filtering to the `patching_strategy` with a mask filter.

    Parameters
    ----------
    patching_strategy : StratifiedPatchingStrategy
        A stratified patching strategy to filter.
    mask_filter_config : MaskFilterConfig
        Configuration for filtering with a mask.
    mask_extractor : PatchExtractor[ImageStack]
        A patch extractor holding the mask data.
    bg_relative_prob : float
        The probability that a region determined as background will be sampled from each
        epoch.
    """
    mask_filter = MaskFilter(coverage=mask_filter_config.coverage)
    patch_size = patching_strategy.patch_size
    region_size = tuple(ps * 2 for ps in patch_size)
    all_grid_coords = patching_strategy.get_all_grid_coords().items()
    n_patches = patching_strategy.n_patches
    n_filtered = 0
    for (data_idx, sample_idx), grid_coords in tqdm(
        all_grid_coords, desc="Filtering background patches with provided mask."
    ):
        probs: dict[tuple[int, ...], float] = {}
        for coords in grid_coords:
            region_coords = tuple(
                ps * c for ps, c in zip(patch_size, coords, strict=True)
            )
            mask_patch = mask_extractor.extract_patch(
                data_idx=data_idx,
                sample_idx=sample_idx,
                coords=region_coords,
                patch_size=region_size,
            )
            if mask_filter.filter_out(mask_patch):
                probs[coords] = bg_relative_prob
                n_filtered += 1
        patching_strategy.set_region_probs(data_idx, sample_idx, probs)
    reduced_patches = patching_strategy.n_patches
    logger.info(
        f"Found {n_filtered} background regions. Number of patches has been reduced to "
        f"{reduced_patches} from {n_patches}."
    )

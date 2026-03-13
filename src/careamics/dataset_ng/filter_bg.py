from tqdm import tqdm

from careamics.utils import get_logger

from .patch_extractor import PatchExtractor
from .patch_filter import MaskFilter, PatchFilterProtocol
from .patching_strategies import PatchSpecs, StratifiedPatchingStrategy

logger = get_logger("Patch filtering")


def filter_background(
    patching_strategy: StratifiedPatchingStrategy,
    input_extractor: PatchExtractor,
    patch_filter: PatchFilterProtocol,
    bg_relative_prob: float = 0.1,
) -> None:
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
            patch = input_extractor.extract_patch(
                data_idx,
                sample_idx=sample_idx,
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
    mask_filter: MaskFilter,
    bg_relative_prob: float = 0.1,
) -> None:
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
            region_specs = PatchSpecs(
                data_idx=data_idx,
                sample_idx=sample_idx,
                coords=region_coords,
                patch_size=region_size,
            )
            if mask_filter.filter_out(region_specs):
                probs[coords] = bg_relative_prob
                n_filtered += 1
        patching_strategy.set_region_probs(data_idx, sample_idx, probs)
    reduced_patches = patching_strategy.n_patches
    logger.info(
        f"Found {n_filtered} background regions. Number of patches has been reduced to "
        f"{reduced_patches} from {n_patches}."
    )

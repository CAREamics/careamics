import numpy as np
from tqdm import tqdm

from .patch_extractor import PatchExtractor
from .patch_filter.patch_filter_protocol import PatchFilterProtocol
from .patching_strategies import StratifiedPatchingStrategy


def filter_background(
    patching_strategy: StratifiedPatchingStrategy,
    input_extractor: PatchExtractor,
    patch_filter: PatchFilterProtocol,
    ref_channel: int,
    bg_relative_prob: float = 0.1,
) -> None:
    patch_size = patching_strategy.patch_size
    region_size = tuple(ps * 2 for ps in patch_size)
    all_grid_coords = patching_strategy.get_all_grid_coords().items()
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
        patching_strategy.set_region_probs(data_idx, sample_idx, probs)


def filter_background_with_mask(
    patching_strategy: StratifiedPatchingStrategy,
    mask_extractor: PatchExtractor,
    threshold_ratio: float,
    bg_relative_prob: float = 0.1,
) -> None:
    patch_size = patching_strategy.patch_size
    region_size = tuple(ps * 2 for ps in patch_size)
    all_grid_coords = patching_strategy.get_all_grid_coords().items()
    for (data_idx, sample_idx), grid_coords in tqdm(
        all_grid_coords, desc="Filtering background patches with provided mask."
    ):
        probs: dict[tuple[int, ...], float] = {}
        for coords in grid_coords:
            region_coords = tuple(
                ps * c for ps, c in zip(patch_size, coords, strict=True)
            )
            mask_patch = mask_extractor.extract_channel_patch(
                data_idx,
                sample_idx=sample_idx,
                channels=None,
                coords=region_coords,
                patch_size=region_size,
            )
            mask_patch = mask_patch.astype(bool)
            if np.mean(mask_patch) < threshold_ratio:
                probs[coords] = bg_relative_prob
        patching_strategy.set_region_probs(data_idx, sample_idx, probs)

import numpy as np

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
    all_grid_coords = patching_strategy.get_all_grid_coords().items()
    for (data_idx, sample_idx), grid_coords in all_grid_coords:
        probs: dict[tuple[int, int], float] = {}
        for coords in grid_coords:
            spatial_shape = input_extractor.shapes[data_idx][2:]
            patch_coords = tuple(
                ps * c for ps, c in zip(patch_size, coords, strict=True)
            )
            patch_end = tuple(
                np.clip(pc + ps, 0, ss)
                for pc, ps, ss in zip(
                    patch_coords, patch_size, spatial_shape, strict=True
                )
            )
            patch_size_clipped = tuple(
                pe - pc for pe, pc in zip(patch_end, patch_coords, strict=True)
            )
            patch = input_extractor.extract_channel_patch(
                data_idx,
                sample_idx=sample_idx,
                channels=[ref_channel],
                coords=patch_coords,
                patch_size=patch_size_clipped,
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
    all_grid_coords = patching_strategy.get_all_grid_coords().items()
    for (data_idx, sample_idx), grid_coords in all_grid_coords:
        probs: dict[tuple[int, int], float] = {}
        for coords in grid_coords:
            spatial_shape = mask_extractor.shapes[data_idx][2:]
            patch_coords = tuple(
                ps * c for ps, c in zip(patch_size, coords, strict=True)
            )
            patch_end = tuple(
                np.clip(pc + ps, 0, ss)
                for pc, ps, ss in zip(
                    patch_coords, patch_size, spatial_shape, strict=True
                )
            )
            patch_size_clipped = tuple(
                pe - pc for pe, pc in zip(patch_end, patch_coords, strict=True)
            )
            mask_patch = mask_extractor.extract_channel_patch(
                data_idx,
                sample_idx=sample_idx,
                channels=None,
                coords=patch_coords,
                patch_size=patch_size_clipped,
            )
            mask_patch = mask_patch.astype(bool)
            if np.mean(mask_patch) < threshold_ratio:
                probs[coords] = bg_relative_prob
        patching_strategy.set_region_probs(data_idx, sample_idx, probs)

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from careamics.config.data.normalization_config import (
    MeanStdConfig,
    MinMaxConfig,
    NoNormConfig,
    NormalizationConfig,
    QuantileConfig,
)
from careamics.dataset.dataset_utils.running_stats import WelfordStatistics
from careamics.dataset_ng.normalization.running_quantile import QuantileEstimator
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patching_strategies import PatchingStrategy


def _compute_mean_std(
    data_extractor: PatchExtractor,
    patching_strategy: PatchingStrategy,
    channels: Sequence[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute mean and std from data."""
    image_stats = WelfordStatistics()
    n_patches = patching_strategy.n_patches

    for idx in tqdm(range(n_patches), desc="Computing mean/std statistics"):
        patch_spec = patching_strategy.get_patch_spec(idx)
        patch = data_extractor.extract_channel_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            channels=channels,
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )
        image_stats.update(patch[None, ...], sample_idx=idx)

    means, stds = image_stats.finalize()
    return means.tolist(), stds.tolist()


def _compute_min_max(
    data_extractor: PatchExtractor,
    patching_strategy: PatchingStrategy,
    channels: Sequence[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute min and max from data."""
    n_patches = patching_strategy.n_patches
    if n_patches == 0:
        raise ValueError("No patches found to compute min/max statistics.")

    first_spec = patching_strategy.get_patch_spec(0)
    first_patch = data_extractor.extract_patch(
        data_idx=first_spec["data_idx"],
        sample_idx=first_spec["sample_idx"],
        coords=first_spec["coords"],
        patch_size=first_spec["patch_size"],
    )
    axes = tuple(dim for dim in range(first_patch.ndim) if dim != 0)
    min_vals: NDArray = np.min(first_patch, axis=axes)
    max_vals: NDArray = np.max(first_patch, axis=axes)

    for idx in tqdm(range(1, n_patches), desc="Computing min/max statistics"):
        patch_spec = patching_strategy.get_patch_spec(idx)
        patch = data_extractor.extract_channel_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            channels=channels,
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )
        current_mins = np.min(patch, axis=axes)
        current_maxes = np.max(patch, axis=axes)
        min_vals = np.minimum(min_vals, current_mins)
        max_vals = np.maximum(max_vals, current_maxes)

    return min_vals.tolist(), max_vals.tolist()


def _compute_quantiles(
    data_extractor: PatchExtractor,
    patching_strategy: PatchingStrategy,
    lower_quantiles: list[float],
    upper_quantiles: list[float],
    channels: Sequence[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute quantile values from data."""
    estimator = QuantileEstimator(
        lower_quantiles=lower_quantiles,
        upper_quantiles=upper_quantiles,
    )

    n_patches = patching_strategy.n_patches
    for idx in tqdm(range(n_patches), desc="Computing quantile statistics"):
        patch_spec = patching_strategy.get_patch_spec(idx)
        patch = data_extractor.extract_channel_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            channels=channels,
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )
        estimator.update(patch)

    lower_values, upper_values = estimator.finalize()
    return lower_values.tolist(), upper_values.tolist()


def resolve_normalization_config(
    norm_config: NormalizationConfig,
    patching_strategy: PatchingStrategy,
    input_extractor: PatchExtractor,
    target_extractor: PatchExtractor | None = None,
    channels: Sequence[int] | None = None,
) -> NormalizationConfig:
    """
    Resolve a normalization config by computing any missing statistics.

    If statistics are already provided in the config, they are preserved.
    If statistics are missing (None), they are computed from the data.

    Parameters
    ----------
    norm_config : NormalizationConfig
        The normalization configuration (may have missing statistics).
    patching_strategy : PatchingStrategy
        Strategy for iterating over patches.
    channels : list[int]
        Channels to compute statistics for.
    input_extractor : PatchExtractor
        Extractor for input data.
    target_extractor : PatchExtractor, optional
        Extractor for target data.

    Returns
    -------
    NormalizationConfig
        A resolved configuration with all statistics populated.
    """
    if isinstance(norm_config, NoNormConfig):
        return norm_config

    if isinstance(norm_config, MeanStdConfig):
        if norm_config.needs_computation():
            input_means, input_stds = _compute_mean_std(
                input_extractor, patching_strategy, channels
            )
            norm_config.set_input_stats(input_means, input_stds)

        if target_extractor is not None and norm_config.target_means is None:
            target_means, target_stds = _compute_mean_std(
                target_extractor, patching_strategy, channels
            )
            norm_config.set_target_stats(target_means, target_stds)

        return norm_config

    if isinstance(norm_config, MinMaxConfig):
        if norm_config.needs_computation():
            input_mins, input_maxes = _compute_min_max(
                input_extractor, patching_strategy, channels
            )
            norm_config.set_input_range(input_mins, input_maxes)

        if target_extractor is not None and norm_config.target_mins is None:
            target_mins, target_maxes = _compute_min_max(
                target_extractor, patching_strategy, channels
            )
            norm_config.set_target_range(target_mins, target_maxes)

        return norm_config

    if isinstance(norm_config, QuantileConfig):
        if norm_config.needs_computation():
            first_spec = patching_strategy.get_patch_spec(0)
            first_patch = input_extractor.extract_channel_patch(
                data_idx=first_spec["data_idx"],
                sample_idx=first_spec["sample_idx"],
                channels=channels,
                coords=first_spec["coords"],
                patch_size=first_spec["patch_size"],
            )
            n_channels = first_patch.shape[0]

            lower_levels = norm_config.get_lower_quantiles_for_channels(n_channels)
            upper_levels = norm_config.get_upper_quantiles_for_channels(n_channels)

            lower_values, upper_values = _compute_quantiles(
                input_extractor,
                patching_strategy,
                lower_levels,
                upper_levels,
                channels=channels,
            )
            norm_config.set_input_quantile_values(lower_values, upper_values)

        if (
            target_extractor is not None
            and norm_config.target_lower_quantile_values is None
        ):
            first_spec = patching_strategy.get_patch_spec(0)
            first_patch = target_extractor.extract_channel_patch(
                data_idx=first_spec["data_idx"],
                sample_idx=first_spec["sample_idx"],
                channels=channels,
                coords=first_spec["coords"],
                patch_size=first_spec["patch_size"],
            )
            n_channels = first_patch.shape[0]

            lower_levels = norm_config.get_lower_quantiles_for_channels(n_channels)
            upper_levels = norm_config.get_upper_quantiles_for_channels(n_channels)

            target_lower, target_upper = _compute_quantiles(
                target_extractor,
                patching_strategy,
                lower_levels,
                upper_levels,
                channels=channels,
            )
            norm_config.set_target_quantile_values(target_lower, target_upper)

        return norm_config

    raise ValueError(f"Unknown normalization config type: {type(norm_config)}")

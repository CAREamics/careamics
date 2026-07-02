"""Compute and resolve normalization statistics from patch data."""

from typing import Any, Literal

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
from careamics.dataset.patch_constructor import (
    PatchConstr,
)

from .running_mean_std import WelfordStatistics
from .running_quantile import QuantileEstimator
from .utils import broadcast_stats

StatsDict = dict[Literal["input", "target"], tuple[NDArray[Any], NDArray[Any]]]


def _compute_mean_std(
    patch_constructor: PatchConstr,
    per_channel: bool,
    compute_input: bool,
    compute_target: bool,
) -> StatsDict:
    """Compute mean and std from data.

    Parameters
    ----------
    patch_constructor : PatchConstructor
        Constructor for paired input and target patches.
    per_channel : bool
        If True, computes per-channel statistics. If False, collapse all channels into
        one to produce a single mean/std pair.
    compute_input : bool
        Whether the statistics for the input data should be calculated.
    compute_target : bool
        Whether the statistics for the target data should be calculated. If there is no
        target data present, this argument is ignored.

    Returns
    -------
    dict of str to tuple of NDArray
        A dictionary with an `"input"` entry and, when targets exist, a `"target"`
        entry. Each entry contains the mean and standard deviation values per channel,
        or a single mean/std pair if `per_channel` is False.
    """
    n_patches = patch_constructor.n_patches
    if n_patches == 0:
        raise ValueError("No patches found to compute mean/std statistics.")

    stats: StatsDict = {}
    if not compute_input and not compute_target:
        return stats

    input_stats = WelfordStatistics()
    target_stats = WelfordStatistics()
    for idx in tqdm(range(n_patches), desc="Computing mean/std statistics"):
        input_patch, target_patch, _ = patch_constructor.construct_patch(idx)
        input_patch = patch_constructor.get_principal_input(input_patch)

        if compute_input:
            if not per_channel:
                # flatten along axis for "global" stats
                input_patch = input_patch.reshape(1, -1)
            input_stats.update(input_patch[None, ...], sample_idx=idx)

        if compute_target and target_patch is not None:
            if not per_channel:
                target_patch = target_patch.reshape(1, -1)
            target_stats.update(target_patch[None, ...], sample_idx=idx)

    if compute_input:
        input_means, input_stds = input_stats.finalize()
        stats["input"] = (input_means, input_stds)
    if compute_target:
        target_means, target_stds = target_stats.finalize()
        stats["target"] = target_means, target_stds
    return stats


def _compute_min_max(
    patch_constructor: PatchConstr,
    per_channel: bool,
    compute_input: bool,
    compute_target: bool,
) -> StatsDict:
    """Compute min and max from data.

    Parameters
    ----------
    patch_constructor : PatchConstructor
        Constructor for paired input and target patches.
    per_channel : bool, optional
        If True, computes per-channel statistics.
        If False, collapse all channels into one to produce a single min/max pair.
    compute_input : bool
        Whether the statistics for the input data should be calculated.
    compute_target : bool
        Whether the statistics for the target data should be calculated. If there is no
        target data present, this argument is ignored.

    Returns
    -------
    dict of str to tuple of NDArray
        A dictionary with an `"input"` entry and, when targets exist, a `"target"`
        entry. Each entry contains minimum and maximum values per channel, or a single
        min/max pair if `per_channel` is False.
    """
    n_patches = patch_constructor.n_patches
    if n_patches == 0:
        raise ValueError("No patches found to compute mean/std statistics.")

    stats: StatsDict = {}
    if not compute_input and not compute_target:
        return stats

    input_mins: NDArray | None = None
    input_maxes: NDArray | None = None
    target_mins: NDArray | None = None
    target_maxes: NDArray | None = None
    for idx in tqdm(range(n_patches), desc="Computing min/max statistics"):
        input_patch, target_patch, _ = patch_constructor.construct_patch(idx)
        input_patch = patch_constructor.get_principal_input(input_patch)

        if compute_input:
            if not per_channel:
                input_patch = input_patch.reshape(1, -1)
            current_mins = np.min(input_patch, axis=tuple(range(1, input_patch.ndim)))
            current_maxes = np.max(input_patch, axis=tuple(range(1, input_patch.ndim)))
            input_mins = (
                current_mins
                if input_mins is None
                else np.minimum(input_mins, current_mins)
            )
            input_maxes = (
                current_maxes
                if input_maxes is None
                else np.maximum(input_maxes, current_maxes)
            )

        if compute_target and target_patch is not None:
            if not per_channel:
                target_patch = target_patch.reshape(1, -1)
            current_mins = np.min(target_patch, axis=tuple(range(1, target_patch.ndim)))
            current_maxes = np.max(
                target_patch, axis=tuple(range(1, target_patch.ndim))
            )
            target_mins = (
                current_mins
                if target_mins is None
                else np.minimum(target_mins, current_mins)
            )
            target_maxes = (
                current_maxes
                if target_maxes is None
                else np.maximum(target_maxes, current_maxes)
            )

    if compute_input:
        assert input_mins is not None
        assert input_maxes is not None
        stats["input"] = (input_mins, input_maxes)
    if compute_target:
        assert target_mins is not None
        assert target_maxes is not None
        stats["target"] = (target_mins, target_maxes)
    return stats


def _compute_quantiles(
    patch_constructor: PatchConstr,
    norm_config: QuantileConfig,
    per_channel: bool,
    compute_input: bool,
    compute_target: bool,
) -> StatsDict:
    """Compute quantile values from data.

    Parameters
    ----------
    patch_constructor : PatchConstructor
        Constructor for paired input and target patches.
    norm_config : QuantileConfig
        Quantile normalization configuration.
    per_channel : bool, optional
        If True, computes per-channel statistics.
        If False, collapse all channels into one to produce a single quantile pair.
    compute_input : bool
        Whether the statistics for the input data should be calculated.
    compute_target : bool
        Whether the statistics for the target data should be calculated. If there is no
        target data present, this argument is ignored.

    Returns
    -------
    dict of str to tuple of NDArray
        A dictionary with an `"input"` entry and, when targets exist, a `"target"`
        entry. Each entry contains lower and upper quantile values per channel, or a
        single lower/upper pair if `per_channel` is False.
    """
    n_patches = patch_constructor.n_patches
    if n_patches == 0:
        raise ValueError("No patches found to compute mean/std statistics.")

    stats: StatsDict = {}
    if not compute_input and not compute_target:
        return stats

    input_estimator: QuantileEstimator | None = None
    target_estimator: QuantileEstimator | None = None
    for idx in tqdm(range(n_patches), desc="Computing quantile statistics"):
        input_patch, target_patch, _ = patch_constructor.construct_patch(idx)
        input_patch = patch_constructor.get_principal_input(input_patch)

        if compute_input:
            if not per_channel:
                input_patch = input_patch.reshape(1, -1)

            if input_estimator is None:
                lower_levels, upper_levels = _resolve_quantile_levels(
                    norm_config, input_patch
                )
                input_estimator = QuantileEstimator(
                    lower_quantiles=lower_levels,
                    upper_quantiles=upper_levels,
                )
            input_estimator.update(input_patch)

        if compute_target and target_patch is not None:
            if target_estimator is None:
                lower_levels, upper_levels = _resolve_quantile_levels(
                    norm_config, target_patch
                )
                target_estimator = QuantileEstimator(
                    lower_quantiles=lower_levels,
                    upper_quantiles=upper_levels,
                )
            target_estimator.update(target_patch)

    if compute_input:
        assert input_estimator is not None
        input_lower, input_upper = input_estimator.finalize()
        stats["input"] = (input_lower, input_upper)
    if compute_target:
        assert target_estimator is not None
        target_lower, target_upper = target_estimator.finalize()
        stats["target"] = target_lower, target_upper
    return stats


def _resolve_quantile_levels(
    norm_config: QuantileConfig,
    patch: NDArray,
) -> tuple[list[float], list[float]]:
    """Get quantile levels, broadcasting to n_channels if `per_channel` is True.

    When `per_channel=False` the stored quantile levels (length 1) are
    returned directly. Otherwise a sample patch is extracted to determine
    the number of channels and the levels are broadcast accordingly.

    Parameters
    ----------
    norm_config : QuantileConfig
        Quantile normalization config with lower/upper quantile levels.
    patch : NDArray
        Patch used to determine the number of channels.

    Returns
    -------
    list of float
        Lower quantile levels.
    list of float
        Upper quantile levels.
    """
    if not norm_config.per_channel:
        return norm_config.lower_quantiles, norm_config.upper_quantiles

    n_channels = patch.shape[0]
    return (
        broadcast_stats(norm_config.lower_quantiles, n_channels, "lower_quantiles"),
        broadcast_stats(norm_config.upper_quantiles, n_channels, "upper_quantiles"),
    )


def resolve_normalization_config(
    norm_config: NormalizationConfig,
    patch_constructor: PatchConstr,
) -> NormalizationConfig:
    """
    Resolve a normalization config by computing any missing statistics.

    If statistics are already provided in the config, they are preserved.
    If statistics are missing (None), they are computed from the data.

    Parameters
    ----------
    norm_config : NormalizationConfig
        The normalization configuration (may have missing statistics).
    patch_constructor : PatchConstructor
        Constructor for paired input and target patches.

    Returns
    -------
    NormalizationConfig
        A resolved configuration with all statistics populated.
    """
    has_target = patch_constructor.target_shapes is not None
    if isinstance(norm_config, NoNormConfig):
        return norm_config

    elif isinstance(norm_config, MeanStdConfig):
        compute_input = norm_config.needs_computation()
        compute_target = has_target and norm_config.target_needs_computation()

        if compute_input or compute_target:
            stats = _compute_mean_std(
                patch_constructor,
                per_channel=norm_config.per_channel,
                compute_input=compute_input,
                compute_target=compute_target,
            )

            if compute_input:
                input_means, input_stds = stats["input"]
                norm_config.set_input_stats(input_means.tolist(), input_stds.tolist())

            if compute_target and "target" in stats:
                target_means, target_stds = stats["target"]
                norm_config.set_target_stats(
                    target_means.tolist(), target_stds.tolist()
                )

        return norm_config

    elif isinstance(norm_config, MinMaxConfig):
        compute_input = norm_config.needs_computation()
        compute_target = has_target and norm_config.target_needs_computation()

        if compute_input or compute_target:
            stats = _compute_min_max(
                patch_constructor,
                per_channel=norm_config.per_channel,
                compute_input=compute_input,
                compute_target=compute_target,
            )

            if compute_input:
                input_mins, input_maxes = stats["input"]
                norm_config.set_input_range(input_mins.tolist(), input_maxes.tolist())

            if compute_target and "target" in stats:
                target_mins, target_maxes = stats["target"]
                norm_config.set_target_range(
                    target_mins.tolist(), target_maxes.tolist()
                )

        return norm_config

    elif isinstance(norm_config, QuantileConfig):
        compute_input = norm_config.needs_computation()
        compute_target = has_target and norm_config.target_needs_computation()

        if compute_input or compute_target:
            stats = _compute_quantiles(
                patch_constructor,
                norm_config,
                per_channel=norm_config.per_channel,
                compute_input=compute_input,
                compute_target=compute_target,
            )

            if compute_input:
                lower_values, upper_values = stats["input"]
                norm_config.set_input_quantile_values(
                    lower_values.tolist(), upper_values.tolist()
                )

            if compute_target and "target" in stats:
                target_lower, target_upper = stats["target"]
                norm_config.set_target_quantile_values(
                    target_lower.tolist(), target_upper.tolist()
                )

        return norm_config

    else:
        raise ValueError(f"Unknown normalization config type: {type(norm_config)}")

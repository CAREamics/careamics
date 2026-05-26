"""Functional tests for MicroSplit normalization statistics."""

from typing import Any

import numpy as np
import pytest

from careamics.config.data import MicroSplitDataConfig
from careamics.config.data.normalization_config import (
    MeanStdConfig,
    MinMaxConfig,
    NormalizationConfig,
    QuantileConfig,
)
from careamics.dataset.factory import (
    MicroSplitMultiplexedTargetData,
    MicroSplitPairedData,
    MicroSplitSeparateTargetData,
    create_microsplit_dataset,
)

PATCH_SIZE = (4, 4)
ALPHA_RANGES = [(0.25, 0.25), (0.75, 0.75)]

LOWER_QUANTILE = 2 / 15
UPPER_QUANTILE = 1 - 2 / 15

TARGET_CH_0 = np.arange(16).reshape(1, 4, 4).astype(np.float32)
TARGET_CH_1 = TARGET_CH_0 + 100
TARGET_PATCH = np.concat([TARGET_CH_0, TARGET_CH_1])

WEIGHTED_TARGET_SUM = (
    ALPHA_RANGES[0][0] * TARGET_CH_0 + ALPHA_RANGES[1][0] * TARGET_CH_1
)
REAL_INPUT = TARGET_CH_0 + TARGET_CH_1

SEPARATE_TARGET_CH_0 = TARGET_CH_0
SEPARATE_TARGET_CH_1 = TARGET_CH_1

MicroSplitTrainingData = (
    MicroSplitMultiplexedTargetData[list[np.ndarray]]
    | MicroSplitSeparateTargetData[list[np.ndarray]]
    | MicroSplitPairedData[list[np.ndarray]]
)


def _expected_stats(
    stat_name: str,
    expected_input: np.ndarray,
    expected_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return expected input and target stats for a normalization kind."""
    match stat_name:
        case "mean_std":
            return (
                expected_input.mean(axis=(1, 2)),
                expected_input.std(axis=(1, 2)),
                expected_target.mean(axis=(1, 2)),
                expected_target.std(axis=(1, 2)),
            )
        case "min_max":
            return (
                expected_input.min(axis=(1, 2)),
                expected_input.max(axis=(1, 2)),
                expected_target.min(axis=(1, 2)),
                expected_target.max(axis=(1, 2)),
            )
        case "quantile":
            return (
                np.quantile(expected_input, LOWER_QUANTILE, axis=(1, 2)),
                np.quantile(expected_input, UPPER_QUANTILE, axis=(1, 2)),
                np.quantile(expected_target, LOWER_QUANTILE, axis=(1, 2)),
                np.quantile(expected_target, UPPER_QUANTILE, axis=(1, 2)),
            )
        case _:
            raise ValueError(f"Unknown normalization: {stat_name}.")


def _assert_normalization_stats(
    norm_config: NormalizationConfig,
    stat_name: str,
    expected_input: np.ndarray,
    expected_target: np.ndarray,
) -> None:
    """Assert normalization stats match the expected input and target patches."""
    input_lower, input_upper, target_lower, target_upper = _expected_stats(
        stat_name, expected_input, expected_target
    )
    match norm_config:
        case MeanStdConfig():
            np.testing.assert_allclose(norm_config.input_means, input_lower)
            np.testing.assert_allclose(norm_config.input_stds, input_upper)
            np.testing.assert_allclose(norm_config.target_means, target_lower)
            np.testing.assert_allclose(norm_config.target_stds, target_upper)
        case MinMaxConfig():
            np.testing.assert_allclose(norm_config.input_mins, input_lower)
            np.testing.assert_allclose(norm_config.input_maxes, input_upper)
            np.testing.assert_allclose(norm_config.target_mins, target_lower)
            np.testing.assert_allclose(norm_config.target_maxes, target_upper)
        case QuantileConfig():
            np.testing.assert_allclose(
                norm_config.input_lower_quantile_values, input_lower, rtol=0.02
            )
            np.testing.assert_allclose(
                norm_config.input_upper_quantile_values, input_upper, rtol=0.02
            )
            np.testing.assert_allclose(
                norm_config.target_lower_quantile_values, target_lower, rtol=0.02
            )
            np.testing.assert_allclose(
                norm_config.target_upper_quantile_values, target_upper, rtol=0.02
            )
        case _:
            raise AssertionError("Unexpected normalization config type.")


@pytest.mark.parametrize(
    ("normalization", "stat_name"),
    [
        (
            {"name": "mean_std", "per_channel": True},
            "mean_std",
        ),
        (
            {"name": "min_max", "per_channel": True},
            "min_max",
        ),
        (
            {
                "name": "quantile",
                "per_channel": True,
                "lower_quantiles": [LOWER_QUANTILE],
                "upper_quantiles": [UPPER_QUANTILE],
            },
            "quantile",
        ),
    ],
)
@pytest.mark.parametrize(
    ("data", "alpha_ranges", "expected_input", "expected_target"),
    [
        pytest.param(
            MicroSplitMultiplexedTargetData([TARGET_PATCH]),
            ALPHA_RANGES,
            WEIGHTED_TARGET_SUM,
            TARGET_PATCH,
            id="t1-multiplexed",
        ),
        pytest.param(
            MicroSplitSeparateTargetData(
                [[SEPARATE_TARGET_CH_0], [SEPARATE_TARGET_CH_1]]
            ),
            ALPHA_RANGES,
            WEIGHTED_TARGET_SUM,
            TARGET_PATCH,
            id="t2-separate",
        ),
        pytest.param(
            MicroSplitPairedData(
                input_data=[REAL_INPUT],
                target_data=[TARGET_PATCH],
            ),
            None,
            REAL_INPUT,
            TARGET_PATCH,
            id="t3-paired",
        ),
    ],
)
def test_microsplit_stats_computation(
    normalization: dict[str, Any],
    stat_name: str,
    data: MicroSplitTrainingData,
    alpha_ranges: list[tuple[float, float]] | None,
    expected_input: np.ndarray,
    expected_target: np.ndarray,
) -> None:
    """Test stats compute from principal input and all target channels."""
    config = MicroSplitDataConfig(
        mode="training",
        data_type="array",
        axes="CYX",
        patching={"name": "stratified", "patch_size": PATCH_SIZE, "seed": 42},
        normalization=normalization,
        alpha_ranges=alpha_ranges,
        multiscale_count=2,
    )
    # stats should be updated in the configuration
    create_microsplit_dataset(
        config=config,
        data=data,
        rng=np.random.default_rng(42),
    )

    _assert_normalization_stats(
        config.normalization,
        stat_name,
        expected_input,
        expected_target,
    )

"""Tests for MicroSplit data configuration."""

from collections.abc import Sequence
from typing import Any

import pytest

from careamics.config.data import MicroSplitDataConfig


def _microsplit_config_dict(**kwargs: Any) -> dict[str, Any]:
    """Return minimal MicroSplit config dictionary for tests."""
    return {
        "mode": "training",
        "data_type": "array",
        "axes": "SCYX",
        "patching": {"name": "stratified", "patch_size": (16, 16), "seed": 42},
        "normalization": {"name": "none"},
        **kwargs,
    }


@pytest.mark.parametrize(
    ("invalid_kwargs", "match"),
    [
        (
            {"uncorrelated_channel_prob": 1.0},
            "Spatially uncorrelated channels are not supported for prediction",
        ),
        (
            {"alpha_ranges": [(0.5, 0.5), (0.5, 0.5)]},
            "Alpha ranges cannot be set for prediction",
        ),
    ],
)
def test_predict_rejects_train_options(
    invalid_kwargs: dict[str, Any],
    match: str,
) -> None:
    """Test prediction configs reject MicroSplit options that only train uses."""
    with pytest.raises(ValueError, match=match):
        MicroSplitDataConfig(
            **_microsplit_config_dict(
                mode="predicting",
                patching={"name": "whole"},
                **invalid_kwargs,
            )
        )


def test_patch_filter_is_not_supported() -> None:
    """Test MicroSplit data config rejects patch filtering."""
    with pytest.raises(
        NotImplementedError,
        match="Patch filtering is currently not implemented for MicroSplit",
    ):
        MicroSplitDataConfig(
            **_microsplit_config_dict(
                patch_filter={
                    "name": "max",
                    "threshold": 0.5,
                    "filtered_patch_prob": 0.1,
                }
            )
        )


@pytest.mark.parametrize(
    ("new_mode", "new_patch_size", "overlap_size", "expected_patching"),
    [
        ("validating", None, None, "fixed_random"),
        ("predicting", (16, 16), (8, 8), "tiled"),
    ],
)
def test_convert_mode_preserves_ms_options(
    new_mode: str,
    new_patch_size: Sequence[int] | None,
    overlap_size: Sequence[int] | None,
    expected_patching: str,
) -> None:
    """Test conversion preserves only MicroSplit options used by converted modes."""
    config = MicroSplitDataConfig(
        **_microsplit_config_dict(
            multiscale_count=3,
            padding_mode="wrap",
            alpha_ranges=[(0.2, 0.2), (0.8, 0.8)],
            uncorrelated_channel_prob=1.0,
        )
    )

    converted = config.convert_mode(
        new_mode, new_patch_size=new_patch_size, overlap_size=overlap_size
    )

    assert isinstance(converted, MicroSplitDataConfig)
    assert converted.mode == new_mode
    assert converted.patching.name == expected_patching
    assert converted.multiscale_count == config.multiscale_count
    assert converted.padding_mode == config.padding_mode
    assert converted.alpha_ranges is None
    assert converted.uncorrelated_channel_prob == 0.0

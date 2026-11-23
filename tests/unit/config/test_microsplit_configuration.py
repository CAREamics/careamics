"""Unit tests for MicroSplit configuration."""

from unittest.mock import Mock

import pytest

from careamics.config.factories import create_advanced_microsplit_config


@pytest.mark.parametrize(
    "save_top_k, expected_warning_count",
    [(0, 1), (1, 0)],
)
def test_checkpointing_warning(
    checkpointing_warning: Mock,
    save_top_k,
    expected_warning_count,
):
    """Test checkpointing guidance for MicroSplit."""
    create_advanced_microsplit_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        output_channels=2,
        checkpoint_params={"save_top_k": save_top_k, "save_last": True},
    )

    assert checkpointing_warning.call_count == expected_warning_count

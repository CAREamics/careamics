"""Tests for CryoCARE data configuration."""

import pytest
from pydantic import ValidationError

from careamics.config import CryoCAREDataConfig
from careamics.config.data.normalization_config import NoNormConfig


def _cryocare_data(**kwargs) -> dict:
    data = {
        "mode": "training",
        "data_type": "array",
        "axes": "ZYX",
        "patching": {"name": "stratified", "patch_size": (32, 32, 32)},
        "normalization": NoNormConfig(),
        "batch_size": 1,
    }
    data.update(kwargs)
    return data


@pytest.mark.parametrize(
    "overrides",
    [
        {"axes": "YX"},
        {"axes": "CZYX"},
        {"channels": [0]},
        {"patching": {"name": "random", "patch_size": (32, 32, 32)}},
        {"patching": {"name": "stratified", "patch_size": (32, 32)}},
    ],
)
def test_cryocare_data_constraints(overrides: dict) -> None:
    """Reject data configurations incompatible with CryoCARE."""
    with pytest.raises(ValidationError):
        CryoCAREDataConfig(**_cryocare_data(**overrides))

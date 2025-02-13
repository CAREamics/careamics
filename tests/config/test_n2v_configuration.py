import pytest

from careamics.config import (
    N2VConfiguration,
)
from careamics.config.support import (
    SupportedPixelManipulation,
)


@pytest.mark.parametrize(
    "algorithm, strategy",
    [
        ("n2v", SupportedPixelManipulation.UNIFORM.value),
        ("n2v2", SupportedPixelManipulation.MEDIAN.value),
    ],
)
def test_correct_n2v2_and_transforms(
    minimum_n2v_configuration: dict, algorithm, strategy
):
    """Test that N2V and N2V2 are correctly instantiated."""
    minimum_n2v_configuration["algorithm_config"] = {
        "algorithm": "n2v",
        "loss": "n2v",
        "model": {
            "architecture": "UNet",
            "n2v2": algorithm == "n2v2",
        },
        "n2v_config": {
            "strategy": strategy,
        },
    }

    N2VConfiguration(**minimum_n2v_configuration)


@pytest.mark.parametrize(
    "algorithm, strategy",
    [
        ("n2v", SupportedPixelManipulation.MEDIAN.value),
        ("n2v2", SupportedPixelManipulation.UNIFORM.value),
    ],
)
def test_wrong_n2v2_and_transforms(
    minimum_n2v_configuration: dict, algorithm, strategy
):
    """Test that N2V and N2V2 throw an error if the strategy and the N2V2 UNet
    parameters disagree."""
    minimum_n2v_configuration["algorithm_config"] = {
        "algorithm": "n2v",
        "loss": "n2v",
        "model": {
            "architecture": "UNet",
            "n2v2": algorithm == "n2v2",
        },
    }
    minimum_n2v_configuration["algorithm_config"]["n2v_config"] = [
        {
            "name": "N2VManipulate",
            "strategy": strategy,
        }
    ]

    with pytest.raises(ValueError):
        N2VConfiguration(**minimum_n2v_configuration)

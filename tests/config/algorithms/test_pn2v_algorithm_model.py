import pytest

from careamics.config.algorithms import PN2VAlgorithm


def test_pn2v_requires_noise_model():
    """Check that PN2V algorithm fails without a noise model."""
    model = {
        "architecture": "UNet",
        "in_channels": 1,
        "num_classes": 1,
        "n2v2": False,
    }

    # Test that PN2V algorithm fails without noise model
    with pytest.raises(ValueError):
        PN2VAlgorithm(
            algorithm="pn2v",
            loss="n2v",
            model=model,
            # Missing noise_model parameter
        )

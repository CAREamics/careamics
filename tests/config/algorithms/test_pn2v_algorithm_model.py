import pytest

from careamics.config.algorithms import PN2VAlgorithm
from careamics.config.nm_model import GaussianMixtureNMConfig


def test_pn2v_has_noise_model():
    """Check that PN2V algorithm requires and has a noise model configuration."""
    # Create minimal PN2V configuration with noise model
    noise_model = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path="test_noise_model.npz",
    )

    model = {
        "architecture": "UNet",
        "in_channels": 1,
        "num_classes": 1,
        "n2v2": False,
    }

    # Test that PN2V algorithm can be created with noise model
    pn2v_algo = PN2VAlgorithm(
        algorithm="pn2v",
        loss="n2v",
        model=model,
        noise_model=noise_model,
    )

    # Verify the noise model is present and correctly set
    assert hasattr(pn2v_algo, "noise_model")
    assert pn2v_algo.noise_model is not None
    assert pn2v_algo.noise_model.model_type == "GaussianMixtureNoiseModel"
    assert pn2v_algo.noise_model.path == "test_noise_model.npz"


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

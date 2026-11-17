"""Tests for the PN2V algorithm configuration."""

from careamics.config.algorithms import PN2VAlgorithm


def test_pn2v_algorithm_has_no_noise_model_field():
    """The noise model is a runtime object, not a configuration field."""
    algorithm = PN2VAlgorithm(
        model={"architecture": "UNet", "in_channels": 1, "num_classes": 4}
    )

    assert "noise_model" not in PN2VAlgorithm.model_fields
    assert "noise_model" not in algorithm.model_dump()

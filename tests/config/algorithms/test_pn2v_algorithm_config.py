"""Tests for the PN2V algorithm configuration."""

import pytest

from careamics.config.algorithms import PN2VAlgorithm


def test_pn2v_algorithm_has_no_noise_model_field():
    """The noise model is a runtime object, not a configuration field."""
    algorithm = PN2VAlgorithm(
        model={"architecture": "UNet", "in_channels": 1, "num_classes": 4}
    )

    assert "noise_model" not in PN2VAlgorithm.model_fields
    assert "noise_model" not in algorithm.model_dump()


@pytest.mark.parametrize("independent_channels", [True, False])
@pytest.mark.parametrize(
    "in_channels, num_classes, n_prior_samples", [(1, 2, 2), (1, 800, 800), (2, 8, 4)]
)
def test_pn2v_prior_samples_per_input_channel(
    independent_channels: bool, in_channels: int, num_classes: int, n_prior_samples: int
):
    """`num_classes` is `in_channels` times the number of prior samples."""
    algorithm = PN2VAlgorithm(
        model={
            "architecture": "UNet",
            "in_channels": in_channels,
            "num_classes": num_classes,
            "independent_channels": independent_channels,
        }
    )

    assert algorithm.n_prior_samples == n_prior_samples


@pytest.mark.parametrize(
    "in_channels, num_classes, match",
    [(2, 5, "must be a multiple"), (1, 1, "at least 2"), (2, 2, "at least 2")],
)
def test_pn2v_rejects_invalid_prior_samples(
    in_channels: int, num_classes: int, match: str
):
    """Every input channel needs the same number, at least 2, of prior samples."""
    with pytest.raises(ValueError, match=match):
        PN2VAlgorithm(
            model={
                "architecture": "UNet",
                "in_channels": in_channels,
                "num_classes": num_classes,
            }
        )


def test_pn2v_sample_scale_must_be_positive():
    """The sample scale multiplies the network outputs and must be positive."""
    with pytest.raises(ValueError):
        PN2VAlgorithm(
            model={"architecture": "UNet", "num_classes": 4},
            sample_scale=0.0,
        )


def test_pn2v2_requires_median_strategy():
    """PN2V2 uses the median pixel manipulation, like N2V2."""
    model = {"architecture": "UNet", "num_classes": 4, "n2v2": True}

    with pytest.raises(ValueError, match="median"):
        PN2VAlgorithm(model=model)

    algorithm = PN2VAlgorithm(model=model, n2v_config={"strategy": "median"})
    assert algorithm.get_algorithm_friendly_name() == "PN2V2"

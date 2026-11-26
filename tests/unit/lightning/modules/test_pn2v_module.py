"""Unit tests for the PN2V Lightning module.

The `CAREamist` workflows are tested in `tests/integration/test_pn2v_workflow.py`.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from careamics.config.factories import (
    create_advanced_n2v_config,
    create_advanced_pn2v_config,
)
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.dataset.normalization import (
    MeanStdNormalization,
    NoNormalization,
    RangeNormalization,
)
from careamics.lightning.modules import PN2VModule, create_module
from careamics.lightning.modules.pn2v_module import pn2v_mmse
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)
from tests.utils import gaussian_noise_model_testing


def _pn2v_config(**kwargs):
    """Return a small PN2V configuration for `SYX` arrays."""
    params = {
        "experiment_name": "pn2v_module",
        "data_type": "array",
        "axes": "SYX",
        "patch_size": (16, 16),
        "batch_size": 2,
        "num_epochs": 1,
        "num_steps": 1,
        "n_prior_samples": 2,
        "model_params": {"depth": 2, "num_channels_init": 8},
        "n_val_patches": 1,
        "masked_pixel_percentage": 5,
        "roi_size": 5,
    }
    params.update(kwargs)
    return create_advanced_pn2v_config(**params)


def test_likelihood_broadcasts_over_sample_channels(
    pn2v_noise_model: MultiChannelNoiseModel,
):
    """Observations with one channel score every prior sample."""
    observations = torch.rand(2, 1, 8, 8)
    samples = torch.rand(2, 4, 8, 8)

    likelihoods = pn2v_noise_model.nmodel_0.likelihood(observations, samples)

    assert likelihoods.shape == samples.shape


def test_pn2v_mmse_collapses_sample_axis(pn2v_noise_model: MultiChannelNoiseModel):
    """MMSE is a weighted mean over the sample channel."""
    observations = torch.full((1, 1, 4, 4), 1.0)
    samples = torch.rand(1, 3, 4, 4)

    estimate = pn2v_mmse(samples, observations, pn2v_noise_model.nmodel_0)

    assert estimate.shape == (1, 1, 4, 4)
    assert torch.isfinite(estimate).all()


def test_create_module_returns_pn2v_module():
    """PN2V configs build a PN2VModule with the configured sample scale."""
    config = create_advanced_pn2v_config(
        experiment_name="pn2v_module",
        data_type="array",
        axes="YX",
        patch_size=[32, 32],
        batch_size=1,
        n_prior_samples=2,
        model_params={"depth": 2, "num_channels_init": 8},
    )

    module = create_module(config.algorithm_config)

    assert isinstance(module, PN2VModule)
    assert module.config.sample_scale == 10.0
    output = module(torch.rand(1, 1, 32, 32))
    assert output.shape == (1, 2, 32, 32)


def test_set_noise_model_keeps_it_out_of_state_dict(
    pn2v_noise_model: MultiChannelNoiseModel,
):
    """The noise model is attached, but not registered as a submodule."""
    module = PN2VModule(_pn2v_config().algorithm_config)

    module.set_noise_model(pn2v_noise_model)

    assert module._raw_noise_model is pn2v_noise_model
    assert not any("noise_model" in key for key in module.state_dict())
    assert "_raw_noise_model" not in dict(module.named_modules())


def test_set_noise_model_loads_npz_paths(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
):
    """Per-channel `.npz` paths are accepted, like for HDN and MicroSplit."""
    pn2v_noise_model.nmodel_0.save(str(tmp_path), "nm.npz")
    module = PN2VModule(_pn2v_config().algorithm_config)

    module.set_noise_model([tmp_path / "nm.npz"])

    assert module._raw_noise_model is not None
    assert len(module._raw_noise_model) == 1


def test_set_noise_model_rejects_wrong_channel_count():
    """A noise model needs exactly one channel per input channel."""
    module = PN2VModule(_pn2v_config().algorithm_config)

    with pytest.raises(ValueError, match="one noise model channel per input channel"):
        module.set_noise_model(gaussian_noise_model_testing(n_channels=2))


def test_pn2v_mmse_single_channel_matches_reference(
    pn2v_noise_model: MultiChannelNoiseModel,
):
    """With one channel the MMSE is bitwise identical to the original formula."""
    generator = torch.Generator().manual_seed(0)
    samples = torch.rand(2, 8, 4, 4, generator=generator) * 100
    observations = torch.rand(2, 1, 4, 4, generator=generator) * 100
    gmm = pn2v_noise_model.nmodel_0

    likelihoods = gmm.likelihood(observations, samples)
    expected = torch.sum(likelihoods * samples, dim=1, keepdim=True) / torch.sum(
        likelihoods, dim=1, keepdim=True
    )

    assert torch.equal(pn2v_mmse(samples, observations, gmm), expected)
    assert torch.equal(pn2v_mmse(samples, observations, pn2v_noise_model), expected)


def test_pn2v_mmse_multichannel_is_per_channel():
    """Each channel is the MMSE of its own samples, one output per input channel."""
    generator = torch.Generator().manual_seed(0)
    n_samples = 4
    samples = torch.rand(2, 2 * n_samples, 4, 4, generator=generator) * 100
    observations = torch.rand(2, 2, 4, 4, generator=generator) * 100
    noise_model = gaussian_noise_model_testing(n_channels=2)

    estimate = pn2v_mmse(samples, observations, noise_model)

    assert estimate.shape == (2, 2, 4, 4)
    for c in range(2):
        expected = pn2v_mmse(
            samples[:, c * n_samples : (c + 1) * n_samples],
            observations[:, c : c + 1],
            noise_model.nmodel_0,
        )
        assert torch.equal(estimate[:, c : c + 1], expected)


@pytest.mark.parametrize(
    "normalization",
    [
        MeanStdNormalization(input_means=[12.0], input_stds=[3.5]),
        RangeNormalization(input_mins=[2.0], input_maxes=[250.0]),
        NoNormalization(),
    ],
)
def test_denormalization_matches_normalization(normalization):
    """With one channel, samples and observations are denormalized exactly like the
    `denormalize` method of the dataset normalization."""
    module = PN2VModule(_pn2v_config(n_prior_samples=6).algorithm_config)
    generator = torch.Generator().manual_seed(0)
    prediction = torch.randn(2, 6, 8, 8, generator=generator)
    original = torch.randn(2, 1, 8, 8, generator=generator)

    module._cache_denormalization(normalization)
    samples, observations = module._denormalized_samples(prediction, original)

    expected_samples = normalization.denormalize(prediction * 10.0)
    assert torch.equal(samples, expected_samples)
    assert torch.equal(observations, normalization.denormalize(original))


def test_denormalization_ignores_target_statistics():
    """Samples are predictions of the input, so the input statistics are used even
    when the normalization has target statistics."""
    module = PN2VModule(_pn2v_config().algorithm_config)
    normalization = MeanStdNormalization(
        input_means=[10.0], input_stds=[2.0], target_means=[500.0], target_stds=[50.0]
    )
    original = torch.ones(1, 1, 4, 4)

    module._cache_denormalization(normalization)
    _, observations = module._denormalized_samples(torch.ones(1, 2, 4, 4), original)

    torch.testing.assert_close(observations, original * (2.0 + 1e-6) + 10.0)


@pytest.mark.parametrize("per_channel", [True, False])
def test_denormalization_per_channel(per_channel: bool):
    """Channel `c` and its samples use the statistics of input channel `c`; global
    statistics apply to every channel."""
    config = _pn2v_config(
        axes="SCYX", n_channels=2, n_prior_samples=3, normalization="min_max"
    )
    module = PN2VModule(config.algorithm_config)
    mins, maxes = ([0.0, 100.0], [10.0, 300.0]) if per_channel else ([5.0], [25.0])
    module._cache_denormalization(
        RangeNormalization(input_mins=mins, input_maxes=maxes)
    )

    # normalized value 1 (0.1 times the sample scale of 10) maps to the maximum
    samples, observations = module._denormalized_samples(
        torch.full((1, 6, 4, 4), 0.1), torch.ones(1, 2, 4, 4)
    )

    expected = maxes if per_channel else maxes * 2
    for c in range(2):
        target = torch.tensor(expected[c])
        assert torch.allclose(observations[:, c], target)
        assert torch.allclose(samples[:, 3 * c : 3 * (c + 1)], target)


def test_set_noise_model_multichannel_rejects_single_channel():
    """A two-channel PN2V model needs a two-channel noise model."""
    config = _pn2v_config(axes="SCYX", n_channels=2, n_prior_samples=2)
    module = PN2VModule(config.algorithm_config)

    with pytest.raises(ValueError, match="one noise model channel per input channel"):
        module.set_noise_model(gaussian_noise_model_testing(n_channels=1))

    module.set_noise_model(gaussian_noise_model_testing(n_channels=2))


def test_pn2v_module_rejects_other_algorithms():
    """Like the other UNet modules, a wrong configuration raises a `ValueError`."""
    n2v_config = create_advanced_n2v_config(
        experiment_name="n2v",
        data_type="array",
        axes="YX",
        patch_size=[32, 32],
        batch_size=2,
    )

    with pytest.raises(ValueError, match="must be a PN2VAlgorithm"):
        PN2VModule(n2v_config.algorithm_config)  # type: ignore[arg-type]


def test_noise_model_checkpoint_hooks_round_trip():
    """The checkpoint hooks save the noise model beside the `state_dict` and
    restore it in a new module."""
    config = _pn2v_config(axes="SCYX", n_channels=2)
    noise_model = gaussian_noise_model_testing(n_channels=2)
    module = PN2VModule(config.algorithm_config)
    module.set_noise_model(noise_model)
    checkpoint = {"state_dict": module.state_dict()}

    module.on_save_checkpoint(checkpoint)
    restored = PN2VModule(config.algorithm_config)
    restored.on_load_checkpoint(checkpoint)

    assert not any("noise_model" in key for key in checkpoint["state_dict"])
    assert restored._raw_noise_model is not None
    assert len(restored._raw_noise_model) == 2
    for c in range(2):
        torch.testing.assert_close(
            getattr(restored._raw_noise_model, f"nmodel_{c}").weight,
            getattr(noise_model, f"nmodel_{c}").weight,
        )


def test_pn2v_mmse_known_values(pn2v_noise_model: MultiChannelNoiseModel):
    """With the Gaussian noise model `N(obs; s, 100)`, the MMSE weights each sample
    with its Gaussian likelihood."""
    samples = torch.tensor([0.0, 10.0]).view(1, 2, 1, 1)
    weight = math.exp(-0.5)

    estimate = pn2v_mmse(samples, torch.zeros(1, 1, 1, 1), pn2v_noise_model)
    midpoint = pn2v_mmse(samples, torch.full((1, 1, 1, 1), 5.0), pn2v_noise_model)
    constant = pn2v_mmse(
        torch.full((1, 4, 1, 1), 7.0), torch.zeros(1, 1, 1, 1), pn2v_noise_model
    )

    torch.testing.assert_close(
        estimate, torch.full((1, 1, 1, 1), 10 * weight / (1 + weight))
    )
    torch.testing.assert_close(midpoint, torch.full((1, 1, 1, 1), 5.0))
    torch.testing.assert_close(constant, torch.full((1, 1, 1, 1), 7.0))


def _gmm(seed: int, scale: float) -> GaussianMixtureNoiseModel:
    """Return a 3-Gaussian noise model with fixed weights and a noise scale."""
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.1, (9, 2)).astype(np.float32)
    weight[3:6, 0] = np.log(scale)
    return GaussianMixtureNoiseModel(
        GaussianMixtureNMConfig(
            weight=weight,
            n_gaussian=3,
            n_coeff=2,
            min_signal=0.0,
            max_signal=100.0,
            min_sigma=1.0,
        )
    )


def test_pn2v_mmse_multichannel_uses_channel_noise_models():
    """Channel `c` of the MMSE estimate uses the noise model of channel `c`."""
    generator = torch.Generator().manual_seed(1)
    n_samples = 6
    samples = torch.rand(2, 2 * n_samples, 4, 4, generator=generator) * 100
    observations = torch.rand(2, 2, 4, 4, generator=generator) * 100
    gmms = [_gmm(0, scale=4.0), _gmm(1, scale=400.0)]

    estimate = pn2v_mmse(samples, observations, MultiChannelNoiseModel(gmms))

    for c in range(2):
        channel_samples = samples[:, c * n_samples : (c + 1) * n_samples]
        channel_observations = observations[:, c : c + 1]
        expected = pn2v_mmse(channel_samples, channel_observations, gmms[c])
        swapped = pn2v_mmse(channel_samples, channel_observations, gmms[1 - c])
        assert torch.equal(estimate[:, c : c + 1], expected)
        assert not torch.allclose(estimate[:, c : c + 1], swapped)


def test_independent_channels_outputs_depend_on_own_channel():
    """With independent channels, the samples of channel `c` (outputs
    `[c * K, (c + 1) * K)`) only depend on input channel `c`."""
    n_samples = 3
    config = _pn2v_config(
        axes="SCYX", n_channels=2, n_prior_samples=n_samples, independent_channels=True
    )
    module = PN2VModule(config.algorithm_config).eval()
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 2, 16, 16, generator=generator)

    for changed in range(2):
        x_changed = x.clone()
        x_changed[:, changed] += torch.randn(1, 16, 16, generator=generator)
        with torch.no_grad():
            reference, perturbed = module.model(x), module.model(x_changed)

        for c in range(2):
            outputs = slice(c * n_samples, (c + 1) * n_samples)
            if c == changed:
                assert not torch.allclose(reference[:, outputs], perturbed[:, outputs])
            else:
                assert torch.equal(reference[:, outputs], perturbed[:, outputs])

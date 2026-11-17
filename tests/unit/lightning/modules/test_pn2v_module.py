"""Tests for the PN2V Lightning module."""

from pathlib import Path

import numpy as np
import pytest
import torch

from careamics import CAREamist
from careamics.config.factories import create_advanced_pn2v_config
from careamics.dataset.normalization import (
    MeanStdNormalization,
    NoNormalization,
    RangeNormalization,
)
from careamics.lightning.modules import PN2VModule, create_module
from careamics.lightning.modules.pn2v_module import pn2v_mmse
from careamics.models.lvae.noise_models import MultiChannelNoiseModel
from tests.utils import gaussian_noise_model_testing


def _pn2v_config(tmp_name: str = "pn2v_module", **kwargs):
    """Return a small PN2V configuration for training on `SYX` arrays."""
    params = {
        "experiment_name": tmp_name,
        "data_type": "array",
        "axes": "SYX",
        "patch_size": (16, 16),
        "batch_size": 2,
        "num_epochs": 1,
        "num_steps": 1,
        "num_classes": 2,
        "model_params": {
            "depth": 2,
            "num_channels_init": 8,
            "use_batch_norm": False,
        },
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
        num_classes=2,
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


def test_fit_without_noise_model_raises(tmp_path: Path):
    """Training without a noise model fails before the first step."""
    data = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32) * 100
    engine = CAREamist(_pn2v_config(), work_dir=tmp_path)

    with pytest.raises(ValueError, match="PN2V requires a noise model"):
        engine.train(train_data=data)


def test_pn2v_train_and_predict_smoke(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
) -> None:
    """One training step and a prediction return an MMSE image."""
    data = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32) * 100
    engine = CAREamist(_pn2v_config("pn2v_smoke"), work_dir=tmp_path)

    engine.train(train_data=data, noise_model=pn2v_noise_model)
    predicted, _ = engine.predict(data)

    assert predicted[0].shape[0] == data.shape[0]
    assert predicted[0].shape[-2:] == data.shape[-2:]
    assert np.isfinite(predicted[0]).all()


def test_checkpoint_round_trip_restores_noise_model(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
) -> None:
    """The noise model is saved beside the weights and restored on load."""
    data = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32) * 100
    engine = CAREamist(_pn2v_config("pn2v_ckpt"), work_dir=tmp_path)
    engine.train(train_data=data, noise_model=pn2v_noise_model)
    last = engine.get_checkpoints()[-1]

    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    loaded = PN2VModule.load_from_checkpoint(last, map_location="cpu")

    assert "noise_model" in checkpoint
    assert not any("noise_model" in key for key in checkpoint["state_dict"])
    assert loaded._raw_noise_model is not None
    torch.testing.assert_close(
        loaded._raw_noise_model.nmodel_0.weight.cpu(),
        pn2v_noise_model.nmodel_0.weight.cpu(),
    )
    for key, value in engine.model.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value.cpu())

    expected, _ = engine.predict(data)
    reloaded, _ = CAREamist(checkpoint_path=last, work_dir=tmp_path).predict(data)
    np.testing.assert_allclose(reloaded[0], expected[0], rtol=1e-5, atol=1e-4)


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
    module = PN2VModule(_pn2v_config(num_classes=6).algorithm_config)
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
        axes="SCYX", n_channels=2, num_classes=6, normalization="min_max"
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
    config = _pn2v_config(axes="SCYX", n_channels=2, num_classes=4)
    module = PN2VModule(config.algorithm_config)

    with pytest.raises(ValueError, match="one noise model channel per input channel"):
        module.set_noise_model(gaussian_noise_model_testing(n_channels=1))

    module.set_noise_model(gaussian_noise_model_testing(n_channels=2))


@pytest.mark.parametrize("independent_channels", [True, False])
def test_pn2v_multichannel_train_and_predict(
    tmp_path: Path, independent_channels: bool
) -> None:
    """Two input channels train and predict one MMSE image per channel."""
    rng = np.random.default_rng(0)
    data = rng.random((2, 2, 32, 32), dtype=np.float32) * 100
    data[:, 1] += 1000
    config = _pn2v_config(
        "pn2v_multichannel",
        axes="SCYX",
        n_channels=2,
        num_classes=8,
        independent_channels=independent_channels,
    )
    engine = CAREamist(config, work_dir=tmp_path)

    engine.train(
        train_data=data, noise_model=gaussian_noise_model_testing(n_channels=2)
    )
    predicted, _ = engine.predict(data)

    assert engine.model.model(torch.rand(1, 2, 16, 16)).shape == (1, 8, 16, 16)
    assert predicted[0].shape == data.shape
    assert np.isfinite(predicted[0]).all()
    # each channel is predicted in the intensity range of its own input channel
    assert predicted[0][:, 1].mean() > predicted[0][:, 0].mean() + 500


def test_careamist_train_noise_model_attaches_to_pn2v(tmp_path: Path) -> None:
    """`CAREamist.train_noise_model` fits a PN2V noise model and attaches it."""
    rng = np.random.default_rng(0)
    signal = rng.uniform(0, 255, (4, 16, 16)).astype(np.float32)
    observation = signal + rng.normal(0, 10, signal.shape).astype(np.float32)
    engine = CAREamist(_pn2v_config(), work_dir=tmp_path)

    noise_model = engine.train_noise_model(
        signal,
        observation,
        axes="SYX",
        n_gaussian=1,
        n_coeff=2,
        n_epochs=2,
        save=False,
        lower_clip=0.5,
        upper_clip=99.5,
    )

    assert engine.model._raw_noise_model is noise_model
    expected = np.float32(np.percentile(signal, [0.5, 99.5]))
    np.testing.assert_array_equal(
        [
            noise_model.nmodel_0.min_signal.item(),
            noise_model.nmodel_0.max_signal.item(),
        ],
        expected,
    )

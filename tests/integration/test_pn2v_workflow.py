"""Integration tests for the PN2V workflows through `CAREamist`.

The noise model is a runtime artifact, not part of the configuration. It is passed
to `CAREamist.train(noise_model=...)` or fitted with `CAREamist.train_noise_model`,
attached to `PN2VModule` with `set_noise_model`, and saved in the checkpoints
outside the `state_dict`. Predictions are MMSE estimates, one per input channel.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from careamics import CAREamist
from careamics.config.factories import create_advanced_pn2v_config
from careamics.config.pn2v_configuration import PN2VConfiguration
from careamics.lightning.modules import PN2VModule
from careamics.models.lvae.noise_models import MultiChannelNoiseModel
from tests.utils import gaussian_noise_model_testing


def _pn2v_config(experiment_name: str = "pn2v", **kwargs) -> PN2VConfiguration:
    """Return a small PN2V configuration for training on `SYX` arrays."""
    params = {
        "experiment_name": experiment_name,
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


def _data(shape: tuple[int, ...] = (2, 32, 32)) -> np.ndarray:
    """Return a random image stack with values in `[0, 100)`."""
    return np.random.default_rng(0).random(shape, dtype=np.float32) * 100


def test_fit_without_noise_model_raises(tmp_path: Path):
    """Training without a noise model fails before the first step."""
    data = _data()
    engine = CAREamist(_pn2v_config(), work_dir=tmp_path)

    with pytest.raises(ValueError, match="PN2V requires a noise model"):
        engine.train(train_data=data)


def test_pn2v_train_and_predict(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
) -> None:
    """One training step and a prediction return an MMSE image."""
    data = _data()
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
    data = _data()
    engine = CAREamist(_pn2v_config("pn2v_ckpt"), work_dir=tmp_path)
    engine.train(train_data=data, noise_model=pn2v_noise_model)
    last = engine.get_checkpoint_root() / engine.get_checkpoints()["last"]

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


@pytest.mark.parametrize("independent_channels", [True, False])
def test_pn2v_multichannel_train_and_predict(
    tmp_path: Path, independent_channels: bool
) -> None:
    """Two input channels train and predict one MMSE image per channel."""
    data = _data((2, 2, 32, 32))
    data[:, 1] += 1000
    config = _pn2v_config(
        "pn2v_multichannel",
        axes="SCYX",
        n_channels=2,
        n_prior_samples=4,
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


def test_pn2v_checkpoints_ranked_by_val_loss(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
) -> None:
    """PN2V keeps the best checkpoints by `val_loss` and predicts with the best one
    by default."""
    data = _data()
    config = _pn2v_config("pn2v_best", num_epochs=2)
    engine = CAREamist(config, work_dir=tmp_path)

    engine.train(train_data=data, noise_model=pn2v_noise_model)

    assert config.training_config.checkpoint_params["monitor"] == "val_loss"
    best = Path(engine.trainer.checkpoint_callback.best_model_path)
    assert best.is_file() and not best.stem.endswith("_last")
    record = engine.get_checkpoints()
    assert record["selected"] == record["best"] == best.name

    predicted, _ = engine.predict(data)
    from_best, _ = engine.predict(data, checkpoint=best)
    np.testing.assert_array_equal(predicted[0], from_best[0])


def test_pn2v_tiled_predict(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
) -> None:
    """Tiled prediction stitches the MMSE tiles back to the full image."""
    data = _data()
    engine = CAREamist(_pn2v_config("pn2v_tiled"), work_dir=tmp_path)
    engine.train(train_data=data, noise_model=pn2v_noise_model)

    predicted, _ = engine.predict(data, tile_size=(16, 16), tile_overlap=(4, 4))

    assert predicted[0].shape == (2, 32, 32)
    assert np.isfinite(predicted[0]).all()


def _multichannel_engine(
    tmp_path: Path, name: str, noise_model: MultiChannelNoiseModel
) -> tuple[CAREamist, np.ndarray]:
    """Train a 2-channel PN2V engine for one step, return it and its data."""
    data = _data((2, 2, 32, 32))
    data[:, 1] += 1000
    config = _pn2v_config(name, axes="SCYX", n_channels=2, n_prior_samples=4)
    engine = CAREamist(config, work_dir=tmp_path)
    engine.train(train_data=data, noise_model=noise_model)
    return engine, data


def test_pn2v_multichannel_checkpoint_round_trip(tmp_path: Path) -> None:
    """A 2-channel noise model is saved with the weights and restored on load."""
    noise_model = gaussian_noise_model_testing(n_channels=2)
    engine, data = _multichannel_engine(tmp_path, "pn2v_mc_ckpt", noise_model)
    last = engine.get_checkpoint_root() / engine.get_checkpoints()["last"]

    loaded = PN2VModule.load_from_checkpoint(last, map_location="cpu")

    assert len(loaded._raw_noise_model) == 2
    for c in range(2):
        torch.testing.assert_close(
            getattr(loaded._raw_noise_model, f"nmodel_{c}").weight.cpu(),
            getattr(noise_model, f"nmodel_{c}").weight.cpu(),
        )
    expected, _ = engine.predict(data, checkpoint=last)
    reloaded, _ = CAREamist(checkpoint_path=last, work_dir=tmp_path).predict(data)
    np.testing.assert_allclose(reloaded[0], expected[0], rtol=1e-5, atol=1e-4)


def test_pn2v_multichannel_tiled_predict(tmp_path: Path) -> None:
    """Tiled prediction returns one MMSE image per input channel."""
    noise_model = gaussian_noise_model_testing(n_channels=2)
    engine, data = _multichannel_engine(tmp_path, "pn2v_mc_tiled", noise_model)

    tiled, _ = engine.predict(data, tile_size=(16, 16), tile_overlap=(4, 4))
    whole, _ = engine.predict(data)

    assert tiled[0].shape == data.shape
    assert np.isfinite(tiled[0]).all()
    assert tiled[0][:, 1].mean() > tiled[0][:, 0].mean() + 500
    np.testing.assert_allclose(tiled[0].mean(), whole[0].mean(), rtol=0.05)

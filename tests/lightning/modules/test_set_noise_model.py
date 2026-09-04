"""Tests for `set_noise_model` and the noise-model checkpoint round trip.

The noise model was moved out of the MicroSplit/HDN configuration and is now a
runtime, loss-side artifact attached via `MicroSplitModule.set_noise_model` /
`HDNModule.set_noise_model` and kept out of the module `state_dict`.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from careamics.config.factories import (
    create_advanced_hdn_config,
    create_advanced_microsplit_config,
)
from careamics.config.noise_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.lightning.modules.hdn_module import HDNModule
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.models.lvae.noise_models import (
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
)

pytestmark = pytest.mark.lvae


def _multichannel_nm(n_channels: int, seed: int = 0) -> MultiChannelNoiseModel:
    """Build a raw-space multi-channel noise model with `n_channels` channels."""
    rng = np.random.default_rng(seed)
    configs = [
        GaussianMixtureNMConfig(
            n_gaussian=2,
            n_coeff=2,
            min_signal=0.0,
            max_signal=1.0,
            min_sigma=0.1,
            channel_index=i,
            weight=rng.random((6, 2)),
        )
        for i in range(n_channels)
    ]
    nm = multichannel_noise_model_factory(MultiChannelNMConfig(noise_models=configs))
    assert nm is not None
    return nm


def _microsplit_module(
    output_channels: int = 2, noise_model_likelihood_weight: float = 0.9
) -> MicroSplitModule:
    config = create_advanced_microsplit_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=2,
        output_channels=output_channels,
        multiscale_count=1,
        gaussian_likelihood_weight=1.0 - noise_model_likelihood_weight,
        noise_model_likelihood_weight=noise_model_likelihood_weight,
    )
    return MicroSplitModule(config.algorithm_config)


def _hdn_module(use_noise_model: bool) -> HDNModule:
    config = create_advanced_hdn_config(
        experiment_name="test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=2,
        use_noise_model=use_noise_model,
    )
    return HDNModule(config.algorithm_config)


# --- MicroSplitModule.set_noise_model ---------------------------------------


def test_microsplit_set_noise_model_attaches_raw_model():
    module = _microsplit_module(output_channels=2)
    nm = _multichannel_nm(2)

    module.set_noise_model(nm)

    assert module._raw_noise_model is nm


def test_microsplit_set_noise_model_channel_mismatch_raises():
    module = _microsplit_module(output_channels=2)
    nm = _multichannel_nm(1)

    with pytest.raises(ValueError, match="channel"):
        module.set_noise_model(nm)


def test_microsplit_set_noise_model_warns_when_unused():
    module = _microsplit_module(output_channels=2, noise_model_likelihood_weight=0.0)
    nm = _multichannel_nm(2)

    with pytest.warns(UserWarning, match="will not be used"):
        module.set_noise_model(nm)


def test_microsplit_set_noise_model_not_in_state_dict():
    module = _microsplit_module(output_channels=2)
    module.set_noise_model(_multichannel_nm(2))

    assert not any("noise_model" in key for key in module.state_dict())


# --- HDNModule.set_noise_model -----------------------------------------------


def test_hdn_set_noise_model_attaches_raw_model():
    module = _hdn_module(use_noise_model=True)
    nm = _multichannel_nm(1)

    module.set_noise_model(nm)

    assert module._raw_noise_model is nm


def test_hdn_set_noise_model_warns_when_gaussian_likelihood_selected():
    module = _hdn_module(use_noise_model=False)
    nm = _multichannel_nm(1)

    with pytest.warns(UserWarning, match="will not be used"):
        module.set_noise_model(nm)


def test_hdn_set_noise_model_not_in_state_dict():
    module = _hdn_module(use_noise_model=True)
    module.set_noise_model(_multichannel_nm(1))

    assert not any("noise_model" in key for key in module.state_dict())


# --- MultiChannelNoiseModel.from_npz / to_config -----------------------------


def test_from_npz_roundtrip(tmp_path: Path):
    nm = _multichannel_nm(2)
    nm.nmodel_0.save(str(tmp_path), "ch0.npz", channel_index=0)
    nm.nmodel_1.save(str(tmp_path), "ch1.npz", channel_index=1)

    loaded = MultiChannelNoiseModel.from_npz(
        [tmp_path / "ch0.npz", tmp_path / "ch1.npz"]
    )

    assert len(loaded) == 2


def test_set_noise_model_accepts_npz_paths(tmp_path: Path):
    nm = _multichannel_nm(2)
    nm.nmodel_0.save(str(tmp_path), "ch0.npz", channel_index=0)
    nm.nmodel_1.save(str(tmp_path), "ch1.npz", channel_index=1)

    module = _microsplit_module(output_channels=2)
    module.set_noise_model([tmp_path / "ch0.npz", tmp_path / "ch1.npz"])

    assert len(module._raw_noise_model) == 2


def test_to_config_roundtrip_reproduces_weights():
    nm = _multichannel_nm(2)

    rebuilt = multichannel_noise_model_factory(nm.to_config())

    assert rebuilt is not None
    torch.testing.assert_close(rebuilt.nmodel_0.weight, nm.nmodel_0.weight)
    torch.testing.assert_close(rebuilt.nmodel_1.weight, nm.nmodel_1.weight)


# --- checkpoint round trip ----------------------------------------------------


def _save_and_load_checkpoint(
    module: MicroSplitModule | HDNModule, path: Path
) -> MicroSplitModule | HDNModule:
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": "2.0.0",
    }
    module.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, path)
    return type(module).load_from_checkpoint(path, map_location="cpu")


def test_microsplit_checkpoint_roundtrip_with_noise_model(tmp_path: Path):
    module = _microsplit_module(output_channels=2)
    module.set_noise_model(_multichannel_nm(2))
    ckpt_path = tmp_path / "model.ckpt"

    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": "2.0.0",
    }
    module.on_save_checkpoint(checkpoint)
    assert "noise_model" in checkpoint
    torch.save(checkpoint, ckpt_path)

    restored = MicroSplitModule.load_from_checkpoint(ckpt_path, map_location="cpu")

    assert restored._raw_noise_model is not None
    assert len(restored._raw_noise_model) == 2


def test_microsplit_checkpoint_without_noise_model_has_no_key(tmp_path: Path):
    module = _microsplit_module(output_channels=2, noise_model_likelihood_weight=0.0)
    ckpt_path = tmp_path / "model.ckpt"

    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": "2.0.0",
    }
    module.on_save_checkpoint(checkpoint)
    assert "noise_model" not in checkpoint
    torch.save(checkpoint, ckpt_path)

    restored = MicroSplitModule.load_from_checkpoint(ckpt_path, map_location="cpu")

    assert restored._raw_noise_model is None

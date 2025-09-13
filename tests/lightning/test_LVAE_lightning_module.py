from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from careamics.config import VAEBasedAlgorithm
from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.loss_model import LVAELossConfig
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.lightning import VAEModule
from careamics.losses import (
    denoisplit_loss,
    denoisplit_musplit_loss,
    hdn_loss,
    musplit_loss,
)
from careamics.models.lvae.likelihoods import GaussianLikelihood, NoiseModelLikelihood
from careamics.models.lvae.noise_models import (
    MultiChannelNoiseModel,
)
from careamics.utils.metrics import RunningPSNR

pytestmark = pytest.mark.lvae


# TODO: move to conftest.py as pytest.fixture
def create_dummy_noise_model(
    tmp_path: Path,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> None:
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    np.savez(tmp_path / "dummy_noise_model.npz", **nm_dict)


# TODO: move to conftest.py as pytest.fixture
# it can be split into modules for more clarity (?)
def create_vae_lightning_model(
    tmp_path: Path,
    algorithm: str,
    loss_type: str,
    ll_type: str = "gaussian",  # TODO revisit
    multiscale_count: int = 1,
    predict_logvar: Literal["pixelwise"] | None = None,
    target_ch: int = 1,
) -> VAEModule:
    """Instantiate the muSplit lightining model."""
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
    )

    loss_config = LVAELossConfig(loss_type=loss_type)

    # gaussian likelihood
    if loss_type in ["musplit", "denoisplit_musplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=0.0,
        )
    elif loss_type == "hdn":
        if ll_type == "gaussian":
            gaussian_lik_config = GaussianLikelihoodConfig(
                predict_logvar=predict_logvar,
                logvar_lowerbound=0.0,
            )
        elif ll_type == "nm":
            gaussian_lik_config = None
    else:
        gaussian_lik_config = None

    # noise model likelihood
    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        create_dummy_noise_model(tmp_path, 3, 3)
        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=tmp_path / "dummy_noise_model.npz",
        )
        noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
        nm_lik_config = NMLikelihoodConfig()
    elif loss_type == "hdn":
        if ll_type == "gaussian":
            noise_model_config = None
            nm_lik_config = None
        elif ll_type == "nm":
            create_dummy_noise_model(tmp_path, 3, 3)
            gmm = GaussianMixtureNMConfig(
                model_type="GaussianMixtureNoiseModel",
                path=tmp_path / "dummy_noise_model.npz",
            )
            noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
            nm_lik_config = NMLikelihoodConfig()
    else:
        noise_model_config = None
        nm_lik_config = None

    vae_config = VAEBasedAlgorithm(
        algorithm=algorithm,
        loss=loss_config,
        model=lvae_config,
        gaussian_likelihood=gaussian_lik_config,
        noise_model=noise_model_config,
        noise_model_likelihood=nm_lik_config,
        is_supervised=algorithm != "hdn",
    )

    return VAEModule(
        algorithm_config=vae_config,
    )


# TODO move to conftest.py as a fixture?
class DummyDataset(Dataset):
    def __init__(
        self,
        img_size: int = 64,
        target_ch: int = 1,
        multiscale_count: int = 1,
    ):
        self.num_samples = 3
        self.img_size = img_size
        self.target_ch = target_ch
        self.multiscale_count = multiscale_count

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        input_ = torch.randn(self.multiscale_count, self.img_size, self.img_size)
        target = torch.randn(self.target_ch, self.img_size, self.img_size)
        return input_, target


def create_dummy_dloader(
    batch_size: int = 1,
    img_size: int = 64,
    target_ch: int = 1,
    multiscale_count: int = 1,
):
    dataset = DummyDataset(
        img_size=img_size,
        target_ch=target_ch,
        multiscale_count=multiscale_count,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# TODO predict logvar type ?
@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, ll_type, loss_type, output_channels, exp_error",
    [
        (1, None, "gaussian", "hdn", 1, does_not_raise()),
        (1, None, "nm", "hdn", 1, does_not_raise()),
        (1, None, "gaussian", "hdn", 3, pytest.raises(ValueError)),
        (1, "pixelwise", "nm", "hdn", 1, does_not_raise()),
        (5, None, "nm", "hdn", 1, pytest.raises(ValueError)),
        (1, None, "nm", "denoisplit", 1, pytest.raises(ValueError)),
    ],
)
def test_hdn_lightning_init(
    multiscale_count: int,
    predict_logvar: str,
    ll_type: str,
    loss_type: str,
    output_channels: int,
    exp_error: Callable,
):
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=output_channels,
        predict_logvar=predict_logvar,
    )

    if ll_type == "gaussian":
        gm_likelihood_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=0.0,
        )
        nm_likelihood_config = None
    else:
        nm_likelihood_config = NMLikelihoodConfig(predict_logvar=predict_logvar)
        gm_likelihood_config = None

    with exp_error:
        vae_config = VAEBasedAlgorithm(
            algorithm="hdn",
            loss=LVAELossConfig(loss_type=loss_type),
            model=lvae_config,
            gaussian_likelihood=gm_likelihood_config,
            noise_model_likelihood=nm_likelihood_config,
        )
        lightning_model = VAEModule(
            algorithm_config=vae_config,
        )
        assert lightning_model is not None
        assert isinstance(lightning_model.model, torch.nn.Module)
        if ll_type == "gaussian":
            assert lightning_model.noise_model is None
            assert isinstance(lightning_model.gaussian_likelihood, GaussianLikelihood)
        else:
            assert lightning_model.gaussian_likelihood is None
        assert lightning_model.loss_func == hdn_loss


@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, target_ch, nm_cnt, loss_type, exp_error",
    [
        # musplit cases (target_ch=3, no noise model needed)
        (1, None, 3, 0, "musplit", does_not_raise()),
        (1, "pixelwise", 3, 0, "musplit", does_not_raise()),
        (5, None, 3, 0, "musplit", does_not_raise()),
        (5, "pixelwise", 3, 0, "musplit", does_not_raise()),
        # denoisplit cases
        (1, None, 1, 1, "denoisplit", does_not_raise()),
        (5, None, 1, 1, "denoisplit", does_not_raise()),
        (1, None, 3, 3, "denoisplit", does_not_raise()),
        (5, None, 3, 3, "denoisplit", does_not_raise()),
        (1, None, 2, 4, "denoisplit", pytest.raises(ValidationError)),
        (1, "pixelwise", 1, 1, "denoisplit", pytest.raises(ValidationError)),
        # denoisplit_musplit cases
        (1, None, 1, 1, "denoisplit_musplit", does_not_raise()),
        (1, None, 3, 3, "denoisplit_musplit", does_not_raise()),
        (1, "pixelwise", 1, 1, "denoisplit_musplit", does_not_raise()),
        (1, "pixelwise", 3, 3, "denoisplit_musplit", does_not_raise()),
        (5, None, 1, 1, "denoisplit_musplit", does_not_raise()),
        (5, None, 3, 3, "denoisplit_musplit", does_not_raise()),
        (5, "pixelwise", 1, 1, "denoisplit_musplit", does_not_raise()),
        (5, "pixelwise", 3, 3, "denoisplit_musplit", does_not_raise()),
        (5, "pixelwise", 2, 4, "denoisplit_musplit", pytest.raises(ValidationError)),
    ],
)
def test_microsplit_lightning_init(
    tmp_path: Path,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    nm_cnt: int,
    loss_type: str,
    exp_error: Callable,
):
    # Create the model config
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
    )

    # Create the likelihood config(s)
    # gaussian likelihood
    if loss_type in ["musplit", "denoisplit_musplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=0.0,
        )
    else:
        gaussian_lik_config = None

    # noise model likelihood
    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        create_dummy_noise_model(tmp_path, 3, 3)
        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=tmp_path / "dummy_noise_model.npz",
        )
        noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * nm_cnt)
        nm_lik_config = NMLikelihoodConfig()
    else:
        noise_model_config = None
        nm_lik_config = None

    with exp_error:
        vae_config = VAEBasedAlgorithm(
            algorithm="microsplit",
            loss=LVAELossConfig(loss_type=loss_type),
            model=lvae_config,
            gaussian_likelihood=gaussian_lik_config,
            noise_model=noise_model_config,
            noise_model_likelihood=nm_lik_config,
        )
        lightning_model = VAEModule(
            algorithm_config=vae_config,
        )
        assert lightning_model is not None
        assert isinstance(lightning_model.model, torch.nn.Module)

        # Check specific configurations based on loss type
        if loss_type == "musplit":
            assert lightning_model.noise_model is None
            assert lightning_model.noise_model_likelihood is None
            assert isinstance(lightning_model.gaussian_likelihood, GaussianLikelihood)
            assert lightning_model.loss_func == musplit_loss
        elif loss_type == "denoisplit":
            assert isinstance(lightning_model.noise_model, MultiChannelNoiseModel)
            assert isinstance(
                lightning_model.noise_model_likelihood, NoiseModelLikelihood
            )
            assert lightning_model.gaussian_likelihood is None
            assert lightning_model.loss_func == denoisplit_loss
        elif loss_type == "denoisplit_musplit":
            assert isinstance(lightning_model.noise_model, MultiChannelNoiseModel)
            assert isinstance(
                lightning_model.noise_model_likelihood, NoiseModelLikelihood
            )
            assert isinstance(lightning_model.gaussian_likelihood, GaussianLikelihood)
            assert lightning_model.loss_func == denoisplit_musplit_loss


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("ll_type", ["gaussian", "nm"])
def test_hdn_training_step(
    tmp_path: Path,
    batch_size: int,
    ll_type: str,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=tmp_path,
        algorithm="hdn",
        loss_type="hdn",
        ll_type=ll_type,
        multiscale_count=1,
        predict_logvar=None,
        target_ch=1,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=1,
        target_ch=1,
    )

    if ll_type == "nm":
        data_mean = torch.zeros(1, 1, 1, 1)
        data_std = torch.zeros(1, 1, 1, 1)
        lightning_model.set_data_stats(data_mean, data_std)

    batch = next(iter(dloader))
    train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

    # check outputs
    assert train_loss is not None
    assert isinstance(train_loss, dict)
    assert "loss" in train_loss
    assert "reconstruction_loss" in train_loss
    assert "kl_loss" in train_loss


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("ll_type", ["gaussian", "nm"])
def test_hdn_validation_step(
    tmp_path: Path,
    batch_size: int,
    ll_type: str,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=tmp_path,
        algorithm="hdn",
        loss_type="hdn",
        ll_type=ll_type,
        multiscale_count=1,
        predict_logvar=None,
        target_ch=1,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=1,
        target_ch=1,
    )

    if ll_type == "nm":
        data_mean = torch.zeros(1, 1, 1, 1)
        data_std = torch.zeros(1, 1, 1, 1)
        lightning_model.set_data_stats(data_mean, data_std)

    batch = next(iter(dloader))
    lightning_model.validation_step(batch=batch, batch_idx=0)


@pytest.mark.parametrize(
    "batch_size, multiscale_count, predict_logvar, target_ch, loss_type",
    [
        # musplit cases
        (1, 1, None, 3, "musplit"),
        (1, 1, "pixelwise", 3, "musplit"),
        (1, 5, None, 3, "musplit"),
        (1, 5, "pixelwise", 3, "musplit"),
        (8, 1, None, 3, "musplit"),
        (8, 1, "pixelwise", 3, "musplit"),
        (8, 5, None, 3, "musplit"),
        (8, 5, "pixelwise", 3, "musplit"),
        # denoisplit cases
        (8, 1, None, 1, "denoisplit"),
        (8, 5, None, 1, "denoisplit"),
        (8, 1, None, 3, "denoisplit"),
        (8, 5, None, 3, "denoisplit"),
        # denoisplit_musplit cases
        (8, 1, None, 1, "denoisplit_musplit"),
        (8, 1, None, 3, "denoisplit_musplit"),
        (8, 1, "pixelwise", 1, "denoisplit_musplit"),
        (8, 1, "pixelwise", 3, "denoisplit_musplit"),
        (8, 5, None, 1, "denoisplit_musplit"),
        (8, 5, None, 3, "denoisplit_musplit"),
        (8, 5, "pixelwise", 1, "denoisplit_musplit"),
        (8, 5, "pixelwise", 3, "denoisplit_musplit"),
    ],
)
def test_microsplit_training_step(
    tmp_path: Path,
    batch_size: int,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    loss_type: str,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=(
            tmp_path if loss_type in ["denoisplit", "denoisplit_musplit"] else None
        ),
        algorithm="microsplit",
        loss_type=loss_type,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )

    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        data_mean = torch.zeros(1, target_ch, 1, 1)
        data_std = torch.zeros(1, target_ch, 1, 1)
        lightning_model.set_data_stats(data_mean, data_std)

    batch = next(iter(dloader))
    train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

    # check outputs
    assert train_loss is not None
    assert isinstance(train_loss, dict)
    assert "loss" in train_loss
    assert "reconstruction_loss" in train_loss
    assert "kl_loss" in train_loss


@pytest.mark.parametrize(
    "batch_size, multiscale_count, predict_logvar, target_ch, loss_type",
    [
        # musplit cases
        (1, 1, None, 3, "musplit"),
        (1, 1, "pixelwise", 3, "musplit"),
        (1, 5, None, 3, "musplit"),
        (1, 5, "pixelwise", 3, "musplit"),
        (8, 1, None, 3, "musplit"),
        (8, 1, "pixelwise", 3, "musplit"),
        (8, 5, None, 3, "musplit"),
        (8, 5, "pixelwise", 3, "musplit"),
        # denoisplit cases
        (8, 1, None, 1, "denoisplit"),
        (8, 5, None, 1, "denoisplit"),
        (8, 1, None, 3, "denoisplit"),
        (8, 5, None, 3, "denoisplit"),
        # denoisplit_musplit cases
        (8, 1, None, 1, "denoisplit_musplit"),
        (8, 1, None, 3, "denoisplit_musplit"),
        (8, 1, "pixelwise", 1, "denoisplit_musplit"),
        (8, 1, "pixelwise", 3, "denoisplit_musplit"),
        (8, 5, None, 1, "denoisplit_musplit"),
        (8, 5, None, 3, "denoisplit_musplit"),
        (8, 5, "pixelwise", 1, "denoisplit_musplit"),
        (8, 5, "pixelwise", 3, "denoisplit_musplit"),
    ],
)  # TODO refac losses, leave only microsplit
def test_microsplit_validation_step(
    tmp_path: Path,
    batch_size: int,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    loss_type: str,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=(
            tmp_path if loss_type in ["denoisplit", "denoisplit_musplit"] else None
        ),
        algorithm="microsplit",
        loss_type=loss_type,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )

    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        data_mean = torch.zeros(1, target_ch, 1, 1)
        data_std = torch.zeros(1, target_ch, 1, 1)
        lightning_model.set_data_stats(data_mean, data_std)

    batch = next(iter(dloader))
    lightning_model.validation_step(batch=batch, batch_idx=0)
    # NOTE: `validation_step` does not return anything...


@pytest.mark.parametrize("batch_size", [1, 8])
def test_training_loop_hdn(
    tmp_path: Path,
    batch_size: int,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=tmp_path,
        algorithm="hdn",
        loss_type="hdn",
        multiscale_count=1,
        predict_logvar=None,
        target_ch=1,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=1,
        target_ch=1,
    )
    trainer = Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[])

    try:
        trainer.fit(
            model=lightning_model,
            train_dataloaders=dloader,
            val_dataloaders=dloader,
        )
    except Exception as e:
        pytest.fail(f"Training routine failed with exception: {e}")


@pytest.mark.parametrize(
    "batch_size, multiscale_count, predict_logvar, target_ch, loss_type",
    [
        # musplit cases
        (1, 1, None, 3, "musplit"),
        (1, 1, "pixelwise", 3, "musplit"),
        (1, 5, None, 3, "musplit"),
        (1, 5, "pixelwise", 3, "musplit"),
        (8, 1, None, 3, "musplit"),
        (8, 1, "pixelwise", 3, "musplit"),
        (8, 5, None, 3, "musplit"),
        (8, 5, "pixelwise", 3, "musplit"),
        # denoisplit cases
        (1, 1, None, 1, "denoisplit"),
        (4, 1, None, 1, "denoisplit"),
        (1, 1, None, 3, "denoisplit"),
        (4, 1, None, 3, "denoisplit"),
        # denoisplit_musplit cases
        (1, 1, None, 1, "denoisplit_musplit"),
        (1, 1, None, 3, "denoisplit_musplit"),
        (1, 1, "pixelwise", 1, "denoisplit_musplit"),
        (1, 1, "pixelwise", 3, "denoisplit_musplit"),
        (4, 1, None, 1, "denoisplit_musplit"),
        (4, 1, None, 3, "denoisplit_musplit"),
        (4, 1, "pixelwise", 1, "denoisplit_musplit"),
        (4, 1, "pixelwise", 3, "denoisplit_musplit"),
    ],
)
def test_training_loop_microsplit(
    tmp_path: Path,
    batch_size: int,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    loss_type: str,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=(
            tmp_path if loss_type in ["denoisplit", "denoisplit_musplit"] else None
        ),
        algorithm="microsplit",
        loss_type=loss_type,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=batch_size,
        img_size=64,
        multiscale_count=multiscale_count,
        target_ch=target_ch,
    )

    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        data_mean = torch.zeros(1, target_ch, 1, 1)
        data_std = torch.zeros(1, target_ch, 1, 1)
        lightning_model.set_data_stats(data_mean, data_std)

    trainer = Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[])

    try:
        trainer.fit(
            model=lightning_model,
            train_dataloaders=dloader,
            val_dataloaders=dloader,
        )
    except Exception as e:
        pytest.fail(f"Training routine failed with exception: {e}")


@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_get_reconstructed_tensor(
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=None,
        algorithm="microsplit",
        loss_type="musplit",
        multiscale_count=1,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    dloader = create_dummy_dloader(
        batch_size=1,
        img_size=64,
        multiscale_count=1,
        target_ch=target_ch,
    )
    input_, target = next(iter(dloader))
    output = lightning_model(input_)
    rec_img = lightning_model.get_reconstructed_tensor(output)
    assert rec_img.shape == target.shape  # same shape as target


@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_val_PSNR_computation(
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=None,
        algorithm="microsplit",
        loss_type="musplit",
        multiscale_count=1,
        predict_logvar=predict_logvar,
        target_ch=target_ch,
    )
    assert lightning_model.running_psnr is not None
    assert len(lightning_model.running_psnr) == target_ch
    for item in lightning_model.running_psnr:
        assert isinstance(item, RunningPSNR)

    dloader = create_dummy_dloader(
        batch_size=1,
        img_size=64,
        multiscale_count=1,
        target_ch=target_ch,
    )
    input_, target = next(iter(dloader))
    output = lightning_model(input_)

    curr_psnr = lightning_model.compute_val_psnr(output, target)
    assert curr_psnr is not None
    assert len(curr_psnr) == target_ch
    for i in range(target_ch):
        assert lightning_model.running_psnr[i].mse_sum != 0
        assert lightning_model.running_psnr[i].N == 1
        assert lightning_model.running_psnr[i].min is not None
        assert lightning_model.running_psnr[i].max is not None
        assert lightning_model.running_psnr[i].get() is not None
        lightning_model.running_psnr[i].reset()
        assert lightning_model.running_psnr[i].mse_sum == 0
        assert lightning_model.running_psnr[i].N == 0
        assert lightning_model.running_psnr[i].min is None
        assert lightning_model.running_psnr[i].max is None
        assert lightning_model.running_psnr[i].get() is None

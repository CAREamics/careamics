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
from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import (
    GaussianMixtureNMConfig,
    MultiChannelNMConfig,
)
from careamics.lightning import VAEModule
from careamics.losses import (
    hdn_loss,
    microsplit_loss,
)
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
    tmp_path: Path | None,
    algorithm: str,
    loss_type: str,
    ll_type: str = "gaussian",  # for hdn: "gaussian" or "nm"
    multiscale_count: int = 1,
    predict_logvar: Literal["pixelwise"] | None = None,
    target_ch: int = 1,
    musplit_weight: float | None = None,
    denoisplit_weight: float | None = None,
) -> VAEModule:
    """Instantiate the muSplit lightning model.

    Maps legacy loss_type aliases ("musplit", "denoisplit", "denoisplit_musplit")
    to the new "microsplit" loss_type with appropriate musplit/denoisplit weights.
    """
    # Convert predict_logvar from old string/None to bool.
    # musplit always requires predict_logvar=True.
    if loss_type in ("musplit", "denoisplit_musplit"):
        predict_logvar_bool = True
    else:
        predict_logvar_bool = predict_logvar is not None

    # Map legacy loss_type names to new API
    if loss_type == "musplit":
        actual_loss_type = "microsplit"
        mw = musplit_weight if musplit_weight is not None else 1.0
        dw = denoisplit_weight if denoisplit_weight is not None else 0.0
    elif loss_type == "denoisplit":
        actual_loss_type = "microsplit"
        mw = musplit_weight if musplit_weight is not None else 0.0
        dw = denoisplit_weight if denoisplit_weight is not None else 1.0
    elif loss_type == "denoisplit_musplit":
        actual_loss_type = "microsplit"
        mw = musplit_weight if musplit_weight is not None else 1.0
        dw = denoisplit_weight if denoisplit_weight is not None else 1.0
    else:
        actual_loss_type = loss_type
        mw = musplit_weight if musplit_weight is not None else 0.1
        dw = denoisplit_weight if denoisplit_weight is not None else 0.9

    lvae_config = LVAEConfig(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
    )

    if actual_loss_type == "microsplit":
        loss_config = LVAELossConfig(
            loss_type="microsplit",
            musplit_weight=mw,
            denoisplit_weight=dw,
            predict_logvar=predict_logvar_bool,
        )
    else:
        loss_config = LVAELossConfig(
            loss_type=actual_loss_type,
            predict_logvar=predict_logvar_bool,
        )

    # Create noise model config when needed
    needs_nm = dw > 0 or (algorithm == "hdn" and ll_type == "nm")
    if needs_nm and tmp_path is not None:
        create_dummy_noise_model(tmp_path, 3, 3)
        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=tmp_path / "dummy_noise_model.npz",
        )
        noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    else:
        noise_model_config = None

    vae_config = VAEBasedAlgorithm(
        algorithm=algorithm,
        loss=loss_config,
        model=lvae_config,
        noise_model=noise_model_config,
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
@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, ll_type, loss_type, output_channels, exp_error",
    [
        (1, False, "gaussian", "hdn", 1, does_not_raise()),
        (1, False, "nm", "hdn", 1, does_not_raise()),
        (1, False, "gaussian", "hdn", 3, pytest.raises(ValueError)),
        (1, True, "nm", "hdn", 1, does_not_raise()),
        (5, False, "nm", "hdn", 1, pytest.raises(ValueError)),
        (1, False, "nm", "denoisplit", 1, pytest.raises(ValueError)),
    ],
)
def test_hdn_lightning_init(
    tmp_path: Path,
    multiscale_count: int,
    predict_logvar: str,
    ll_type: str,
    loss_type: str,
    output_channels: int,
    exp_error: Callable,
):
    lvae_config = LVAEConfig(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=output_channels,
    )

    if ll_type == "nm":
        create_dummy_noise_model(tmp_path, 3, 3)
        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=tmp_path / "dummy_noise_model.npz",
        )
        noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * output_channels)
    else:
        noise_model_config = None

    with exp_error:
        vae_config = VAEBasedAlgorithm(
            algorithm="hdn",
            loss=LVAELossConfig(loss_type=loss_type, predict_logvar=predict_logvar),
            model=lvae_config,
            noise_model=noise_model_config,
        )
        lightning_model = VAEModule(
            algorithm_config=vae_config,
        )
        assert lightning_model is not None
        assert isinstance(lightning_model.model, torch.nn.Module)
        if ll_type == "gaussian":
            assert lightning_model.noise_model is None
        else:
            assert isinstance(lightning_model.noise_model, MultiChannelNoiseModel)
        assert lightning_model.loss_func == hdn_loss


@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, target_ch, nm_cnt, musplit_w, denoisplit_w, exp_error",
    [
        # musplit cases (target_ch=3, no noise model needed, musplit_weight=1.0, denoisplit_weight=0.0)
        (1, False, 3, 0, 1.0, 0.0, does_not_raise()),
        (1, True, 3, 0, 1.0, 0.0, does_not_raise()),
        (5, False, 3, 0, 1.0, 0.0, does_not_raise()),
        (5, True, 3, 0, 1.0, 0.0, does_not_raise()),
        # denoisplit cases (musplit_weight=0.0, denoisplit_weight=1.0)
        (1, False, 1, 1, 0.0, 1.0, does_not_raise()),
        (5, False, 1, 1, 0.0, 1.0, does_not_raise()),
        (1, False, 3, 3, 0.0, 1.0, does_not_raise()),
        (5, False, 3, 3, 0.0, 1.0, does_not_raise()),
        (1, False, 2, 4, 0.0, 1.0, pytest.raises(ValidationError)),
        # predict_logvar=True with denoisplit-only is now valid (no longer raises)
        (1, True, 1, 1, 0.0, 1.0, does_not_raise()),
        # denoisplit_musplit cases (both weights=1.0)
        (1, False, 1, 1, 1.0, 1.0, does_not_raise()),
        (1, False, 3, 3, 1.0, 1.0, does_not_raise()),
        (1, True, 1, 1, 1.0, 1.0, does_not_raise()),
        (1, True, 3, 3, 1.0, 1.0, does_not_raise()),
        (5, False, 1, 1, 1.0, 1.0, does_not_raise()),
        (5, False, 3, 3, 1.0, 1.0, does_not_raise()),
        (5, True, 1, 1, 1.0, 1.0, does_not_raise()),
        (5, True, 3, 3, 1.0, 1.0, does_not_raise()),
    ],
)
def test_microsplit_lightning_init(
    tmp_path: Path,
    multiscale_count: int,
    predict_logvar: str,
    target_ch: int,
    nm_cnt: int,
    musplit_w: float,
    denoisplit_w: float,
    exp_error: Callable,
):
    # Create the model config
    lvae_config = LVAEConfig(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=target_ch,
        predict_logvar=predict_logvar,
    )

    if denoisplit_w > 0:
        create_dummy_noise_model(tmp_path, 3, 3)
        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=tmp_path / "dummy_noise_model.npz",
        )
        noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * nm_cnt)
    else:
        noise_model_config = None

    with exp_error:
        vae_config = VAEBasedAlgorithm(
            algorithm="microsplit",
            loss=LVAELossConfig(
                loss_type="microsplit",
                musplit_weight=musplit_w,
                denoisplit_weight=denoisplit_w,
                predict_logvar=predict_logvar,
            ),
            model=lvae_config,
            noise_model=noise_model_config,
        )
        lightning_model = VAEModule(
            algorithm_config=vae_config,
        )
        assert lightning_model is not None
        assert isinstance(lightning_model.model, torch.nn.Module)

        # Check specific configurations based on weights
        if musplit_w > 0 and denoisplit_w == 0:
            # musplit case: no noise model
            assert lightning_model.noise_model is None
            assert lightning_model.loss_func == microsplit_loss
        elif musplit_w == 0 and denoisplit_w > 0:
            # denoisplit case: noise model present
            assert isinstance(lightning_model.noise_model, MultiChannelNoiseModel)
            assert lightning_model.loss_func == microsplit_loss
        elif musplit_w > 0 and denoisplit_w > 0:
            # denoisplit_musplit case: noise model present
            assert isinstance(lightning_model.noise_model, MultiChannelNoiseModel)
            assert lightning_model.loss_func == microsplit_loss


@pytest.mark.skip(reason="Needs to be updated")
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


@pytest.mark.skip(reason="Needs to be updated")
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


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize(
    "batch_size, multiscale_count, predict_logvar, target_ch, loss_type",
    [
        # musplit cases
        (1, 1, False, 3, "musplit"),
        (1, 1, True, 3, "musplit"),
        (1, 5, False, 3, "musplit"),
        (1, 5, True, 3, "musplit"),
        (8, 1, False, 3, "musplit"),
        (8, 1, True, 3, "musplit"),
        (8, 5, False, 3, "musplit"),
        (8, 5, "pixelwise", 3, "musplit"),
        # denoisplit cases
        (8, 1, False, 1, "denoisplit"),
        (8, 5, False, 1, "denoisplit"),
        (8, 1, False, 3, "denoisplit"),
        (8, 5, False, 3, "denoisplit"),
        # denoisplit_musplit cases
        (8, 1, False, 1, "denoisplit_musplit"),
        (8, 1, False, 3, "denoisplit_musplit"),
        (8, 1, True, 1, "denoisplit_musplit"),
        (8, 1, True, 3, "denoisplit_musplit"),
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
        (1, 1, False, 3, "musplit"),
        (1, 1, True, 3, "musplit"),
        (1, 5, False, 3, "musplit"),
        (1, 5, True, 3, "musplit"),
        (8, 1, False, 3, "musplit"),
        (8, 1, True, 3, "musplit"),
        (8, 5, False, 3, "musplit"),
        (8, 5, True, 3, "musplit"),
        # denoisplit cases
        (8, 1, False, 1, "denoisplit"),
        (8, 5, False, 1, "denoisplit"),
        (8, 1, False, 3, "denoisplit"),
        (8, 5, False, 3, "denoisplit"),
        # denoisplit_musplit cases
        (8, 1, False, 1, "denoisplit_musplit"),
        (8, 1, False, 3, "denoisplit_musplit"),
        (8, 1, True, 1, "denoisplit_musplit"),
        (8, 1, True, 3, "denoisplit_musplit"),
        (8, 5, False, 1, "denoisplit_musplit"),
        (8, 5, False, 3, "denoisplit_musplit"),
        (8, 5, True, 1, "denoisplit_musplit"),
        (8, 5, True, 3, "denoisplit_musplit"),
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


@pytest.mark.skip(reason="Needs to be updated")
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
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        callbacks=[],
    )

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
        (1, 1, False, 3, "microsplit"),
        (1, 1, True, 3, "musplit"),
        (1, 5, False, 3, "musplit"),
        (1, 5, True, 3, "microsplit"),
        (8, 1, False, 3, "microsplit"),
        (8, 1, True, 3, "microsplit"),
        (8, 5, False, 3, "musplit"),
        (8, 5, True, 3, "musplit"),
        # denoisplit cases
        (1, 1, False, 1, "denoisplit"),
        (4, 1, False, 1, "denoisplit"),
        (1, 1, False, 3, "denoisplit"),
        (4, 1, False, 3, "denoisplit"),
        # denoisplit_musplit cases
        (1, 1, False, 1, "denoisplit_musplit"),
        (1, 1, False, 3, "denoisplit_musplit"),
        (1, 1, True, 1, "denoisplit_musplit"),
        (1, 1, True, 3, "denoisplit_musplit"),
        (4, 1, False, 1, "denoisplit_musplit"),
        (4, 1, False, 3, "denoisplit_musplit"),
        (4, 1, True, 1, "denoisplit_musplit"),
        (4, 1, True, 3, "denoisplit_musplit"),
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

    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        callbacks=[],
    )

    try:
        trainer.fit(
            model=lightning_model,
            train_dataloaders=dloader,
            val_dataloaders=dloader,
        )
    except Exception as e:
        pytest.fail(f"Training routine failed with exception: {e}")


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize("predict_logvar", [False, True])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_get_reconstructed_tensor(
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=None,
        algorithm="microsplit",
        loss_type="microsplit",
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


@pytest.mark.parametrize("predict_logvar", [False, True])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_val_PSNR_computation(
    predict_logvar: str,
    target_ch: int,
):
    lightning_model = create_vae_lightning_model(
        tmp_path=None,
        algorithm="microsplit",
        loss_type="microsplit",
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

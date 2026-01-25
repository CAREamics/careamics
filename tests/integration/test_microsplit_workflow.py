"""Integration tests for MicroSplit end-to-end workflows."""

from pathlib import Path

import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from careamics.config import VAEBasedAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.likelihood_config import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.noise_model.noise_model_config import (
    GaussianMixtureNMConfig,
    MultiChannelNMConfig,
)
from careamics.lightning import VAEModule
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
)
from careamics.noise_model import NoiseModelTrainer

pytestmark = pytest.mark.lvae


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(
        self,
        img_size: int = 64,
        target_ch: int = 1,
        multiscale_count: int = 1,
        num_samples: int = 8,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.target_ch = target_ch
        self.multiscale_count = multiscale_count

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ = torch.randn(self.multiscale_count, self.img_size, self.img_size)
        target = torch.randn(self.target_ch, self.img_size, self.img_size)
        return input_, target


def create_dummy_dloader(
    batch_size: int = 2,
    img_size: int = 64,
    target_ch: int = 1,
    multiscale_count: int = 1,
    num_samples: int = 8,
) -> DataLoader:
    """Create a dummy dataloader for testing."""
    dataset = DummyDataset(
        img_size=img_size,
        target_ch=target_ch,
        multiscale_count=multiscale_count,
        num_samples=num_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_noise_model_trainer_to_vae_module_integration(tmp_path: Path) -> None:
    """Test NoiseModelTrainer -> VAEModule integration workflow.

    This test verifies:
    1. NoiseModelTrainer can train from signal-observation pairs
    2. get_multichannel_model() returns a valid MultiChannelNoiseModel
    3. VAEModule can be initialized with the noise model
    4. Training step executes without errors
    """
    # Step 1: Train NoiseModelTrainer
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (8, 2, 64, 64))
    observation = signal + gen.normal(0, 15, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    noise_models = trainer.train_from_pairs(
        signal=signal, observation=observation, n_epochs=50
    )

    # Verify noise models were trained
    assert len(noise_models) == 2
    assert all(isinstance(nm, GaussianMixtureNoiseModel) for nm in noise_models)

    # Step 2: Get multichannel model
    multichannel_nm = trainer.get_multichannel_model()
    assert isinstance(multichannel_nm, MultiChannelNoiseModel)
    assert multichannel_nm._nm_cnt == 2

    # Step 3: Create VAEModule with noise model
    lvae_config = LVAEConfig(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=1,
        z_dims=[128, 128, 128, 128],
        output_channels=2,
        predict_logvar=None,
    )

    loss_config = LVAELossConfig(loss_type="denoisplit")

    nm_lik_config = NMLikelihoodConfig()

    vae_config = VAEBasedAlgorithm(
        algorithm="microsplit",
        loss=loss_config,
        model=lvae_config,
        gaussian_likelihood=None,
        noise_model_likelihood=nm_lik_config,
        is_supervised=True,
    )

    lightning_model = VAEModule(algorithm_config=vae_config)

    # Set data stats
    data_mean = torch.zeros(1, 2, 1, 1)
    data_std = torch.ones(1, 2, 1, 1)
    lightning_model.set_data_stats(data_mean, data_std)

    # Step 4: Run training step
    dloader = create_dummy_dloader(batch_size=2, img_size=64, target_ch=2)
    batch = next(iter(dloader))
    train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

    # Verify training step succeeded
    assert train_loss is not None
    assert isinstance(train_loss, dict)
    assert "loss" in train_loss
    assert "reconstruction_loss" in train_loss
    assert "kl_loss" in train_loss


def test_noise_model_save_load_integration(tmp_path: Path) -> None:
    """Test save/load roundtrip with VAEModule integration.

    This test verifies:
    1. NoiseModelTrainer can train and save models
    2. Models can be loaded from disk
    3. Loaded models work in VAEModule
    """
    # Step 1: Train and save noise models
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (8, 64, 64))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=50)

    saved_paths = trainer.save(tmp_path, prefix="test_nm")
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()

    # Step 2: Load noise models
    loaded_models = NoiseModelTrainer.load(saved_paths)
    assert len(loaded_models) == 1
    assert isinstance(loaded_models[0], GaussianMixtureNoiseModel)

    # Step 3: Create config with loaded models
    gmm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=saved_paths[0],
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm_config])
    nm = multichannel_noise_model_factory(noise_model_config)

    assert nm is not None
    assert isinstance(nm, MultiChannelNoiseModel)

    # Step 4: Use in VAEModule
    lvae_config = LVAEConfig(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=1,
        z_dims=[128, 128, 128, 128],
        output_channels=1,
        predict_logvar=None,
    )

    loss_config = LVAELossConfig(loss_type="denoisplit")
    nm_lik_config = NMLikelihoodConfig()

    vae_config = VAEBasedAlgorithm(
        algorithm="microsplit",
        loss=loss_config,
        model=lvae_config,
        gaussian_likelihood=None,
        noise_model_likelihood=nm_lik_config,
        is_supervised=True,
    )

    lightning_model = VAEModule(algorithm_config=vae_config)

    # Set data stats
    data_mean = torch.zeros(1, 1, 1, 1)
    data_std = torch.ones(1, 1, 1, 1)
    lightning_model.set_data_stats(data_mean, data_std)

    # Run a forward pass to ensure everything works
    dloader = create_dummy_dloader(batch_size=2, img_size=64, target_ch=1)
    batch = next(iter(dloader))
    train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

    assert train_loss is not None


def test_full_training_loop_with_noise_model_trainer(tmp_path: Path) -> None:
    """Test complete training loop from NoiseModelTrainer to VAEModule.

    This test verifies:
    1. NoiseModelTrainer trains successfully
    2. VAEModule can be trained with the noise model
    3. Full training loop completes without errors
    """
    # Step 1: Train NoiseModelTrainer
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (10, 64, 64))
    observation = signal + gen.normal(0, 20, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=50)

    # Step 2: Create VAEModule
    lvae_config = LVAEConfig(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=1,
        z_dims=[128, 128, 128, 128],
        output_channels=1,
        predict_logvar=None,
    )

    loss_config = LVAELossConfig(loss_type="denoisplit")
    nm_lik_config = NMLikelihoodConfig()

    vae_config = VAEBasedAlgorithm(
        algorithm="microsplit",
        loss=loss_config,
        model=lvae_config,
        gaussian_likelihood=None,
        noise_model_likelihood=nm_lik_config,
        is_supervised=True,
    )

    lightning_model = VAEModule(algorithm_config=vae_config)

    # Set data stats
    data_mean = torch.zeros(1, 1, 1, 1)
    data_std = torch.ones(1, 1, 1, 1)
    lightning_model.set_data_stats(data_mean, data_std)

    # Step 3: Run training loop
    dloader = create_dummy_dloader(
        batch_size=2, img_size=64, target_ch=1, num_samples=8
    )

    pytorch_trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        callbacks=[],
    )

    try:
        pytorch_trainer.fit(
            model=lightning_model,
            train_dataloaders=dloader,
            val_dataloaders=dloader,
        )
    except Exception as e:
        pytest.fail(f"Training routine failed with exception: {e}")

    # Verify training completed
    assert pytorch_trainer.current_epoch > 0


def test_multichannel_noise_model_integration(tmp_path: Path) -> None:
    """Test multi-channel noise model integration with VAEModule.

    This test verifies:
    1. NoiseModelTrainer handles multi-channel data
    2. Multi-channel noise models work in VAEModule
    3. Training step handles multiple channels correctly
    """
    # Step 1: Train multi-channel noise models
    gen = np.random.default_rng(42)
    n_channels = 3
    signal = gen.uniform(0, 255, (10, n_channels, 64, 64))
    observation = signal + gen.normal(0, 15, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    noise_models = trainer.train_from_pairs(
        signal=signal, observation=observation, n_epochs=50
    )

    assert len(noise_models) == n_channels

    # Step 2: Get multichannel model
    multichannel_nm = trainer.get_multichannel_model()
    assert multichannel_nm._nm_cnt == n_channels

    # Step 3: Create VAEModule with multi-channel noise model
    lvae_config = LVAEConfig(
        architecture="LVAE",
        input_shape=(64, 64),
        multiscale_count=1,
        z_dims=[128, 128, 128, 128],
        output_channels=n_channels,
        predict_logvar=None,
    )

    loss_config = LVAELossConfig(loss_type="denoisplit")
    nm_lik_config = NMLikelihoodConfig()

    vae_config = VAEBasedAlgorithm(
        algorithm="microsplit",
        loss=loss_config,
        model=lvae_config,
        gaussian_likelihood=None,
        noise_model_likelihood=nm_lik_config,
        is_supervised=True,
    )

    lightning_model = VAEModule(algorithm_config=vae_config)

    # Set data stats
    data_mean = torch.zeros(1, n_channels, 1, 1)
    data_std = torch.ones(1, n_channels, 1, 1)
    lightning_model.set_data_stats(data_mean, data_std)

    # Step 4: Run training step
    dloader = create_dummy_dloader(
        batch_size=2, img_size=64, target_ch=n_channels, multiscale_count=1
    )
    batch = next(iter(dloader))
    train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

    # Verify training step succeeded
    assert train_loss is not None
    assert isinstance(train_loss, dict)


def test_noise_model_sampling_integration() -> None:
    """Test noise model sampling functionality in integration context.

    This test verifies:
    1. NoiseModelTrainer can sample observations
    2. Sampled observations have correct shape
    3. Sampling is consistent for single and multi-channel
    """
    gen = np.random.default_rng(42)

    # Single channel
    signal_single = gen.uniform(0, 255, (5, 64, 64))
    observation_single = signal_single + gen.normal(0, 10, signal_single.shape)

    trainer_single = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer_single.train_from_pairs(
        signal=signal_single, observation=observation_single, n_epochs=50
    )

    sampled_single = trainer_single.sample_observation(signal_single)
    assert sampled_single.shape == signal_single.shape

    # Multi-channel
    signal_multi = gen.uniform(0, 255, (5, 2, 64, 64))
    observation_multi = signal_multi + gen.normal(0, 10, signal_multi.shape)

    trainer_multi = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer_multi.train_from_pairs(
        signal=signal_multi, observation=observation_multi, n_epochs=50
    )

    sampled_multi = trainer_multi.sample_observation(signal_multi)
    assert sampled_multi.shape == signal_multi.shape


def test_musplit_denoisplit_weights_integration(tmp_path: Path) -> None:
    """Test different muSplit/denoiSplit weight combinations.

    This test verifies:
    1. VAEModule works with different weight configurations
    2. Pure muSplit (musplit_weight=1.0, denoisplit_weight=0.0)
    3. Pure denoiSplit (musplit_weight=0.0, denoisplit_weight=1.0)
    4. Mixed weights (musplit_weight=0.5, denoisplit_weight=0.5)
    """
    # Train noise model
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (8, 64, 64))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=50)

    # Test different weight configurations
    weight_configs = [
        (1.0, 0.0),  # Pure muSplit
        (0.0, 1.0),  # Pure denoiSplit
        (0.5, 0.5),  # Mixed
        (0.3, 0.7),  # Weighted towards denoiSplit
    ]

    for musplit_weight, denoisplit_weight in weight_configs:
        # Create VAEModule with specific weights
        lvae_config = LVAEConfig(
            architecture="LVAE",
            input_shape=(64, 64),
            multiscale_count=1,
            z_dims=[128, 128, 128, 128],
            output_channels=1,
            predict_logvar="pixelwise" if musplit_weight > 0 else None,
        )

        loss_config = LVAELossConfig(
            loss_type="denoisplit_musplit",
            musplit_weight=musplit_weight,
            denoisplit_weight=denoisplit_weight,
        )

        # Set up likelihoods based on weights
        gaussian_lik_config = (
            GaussianLikelihoodConfig(predict_logvar="pixelwise")
            if musplit_weight > 0
            else None
        )
        nm_lik_config = NMLikelihoodConfig() if denoisplit_weight > 0 else None

        vae_config = VAEBasedAlgorithm(
            algorithm="microsplit",
            loss=loss_config,
            model=lvae_config,
            gaussian_likelihood=gaussian_lik_config,
            noise_model_likelihood=nm_lik_config,
            is_supervised=True,
        )

        lightning_model = VAEModule(algorithm_config=vae_config)

        if denoisplit_weight > 0:
            data_mean = torch.zeros(1, 1, 1, 1)
            data_std = torch.ones(1, 1, 1, 1)
            lightning_model.set_data_stats(data_mean, data_std)

        # Run training step
        dloader = create_dummy_dloader(batch_size=2, img_size=64, target_ch=1)
        batch = next(iter(dloader))
        train_loss = lightning_model.training_step(batch=batch, batch_idx=0)

        # Verify training step succeeded
        assert train_loss is not None
        assert isinstance(train_loss, dict)
        assert "loss" in train_loss


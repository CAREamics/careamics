"""End-to-end tests for 1D N2V training with CAREamist."""

import numpy as np
import pytest

from careamics import CAREamist
from careamics.config import create_n2v_configuration


@pytest.fixture
def sample_1d_data():
    """Create sample 1D data for testing."""
    # Create synthetic 1D data (S, C, X)
    np.random.seed(42)
    n_samples = 10
    n_channels = 1
    length = 256

    # Generate noisy 1D signals
    data = np.random.randn(n_samples, n_channels, length).astype(np.float32)
    # Add some structure (smooth signals)
    for i in range(n_samples):
        x = np.linspace(0, 4 * np.pi, length)
        signal = np.sin(x) + 0.5 * np.sin(2 * x)
        data[i, 0, :] = signal + 0.2 * data[i, 0, :]

    return data


@pytest.fixture
def sample_1d_multichannel_data():
    """Create sample 1D multichannel data."""
    np.random.seed(42)
    n_samples = 10
    n_channels = 4
    length = 256

    data = np.random.randn(n_samples, n_channels, length).astype(np.float32)

    # Add different signals to different channels
    for i in range(n_samples):
        x = np.linspace(0, 4 * np.pi, length)
        for c in range(n_channels):
            signal = np.sin((c + 1) * x)
            data[i, c, :] = signal + 0.2 * data[i, c, :]

    return data


class TestCAREamist1D:
    """Test CAREamist with 1D data."""

    def test_n2v_1d_training(self, tmp_path, sample_1d_data):
        """Test basic N2V training with 1D data."""
        # Create N2V configuration for 1D
        config = create_n2v_configuration(
            experiment_name="test_1d_n2v",
            data_type="array",
            axes="SX",
            patch_size=[64],
            batch_size=2,
            num_epochs=2,
        )

        # Create CAREamist instance
        careamist = CAREamist(source=config, work_dir=tmp_path)

        # Train
        careamist.train(train_source=sample_1d_data)

        # Check model exists
        assert careamist.model is not None

    def test_n2v_1d_prediction(self, tmp_path, sample_1d_data):
        """Test N2V prediction with 1D data."""
        config = create_n2v_configuration(
            experiment_name="test_1d_n2v_pred",
            data_type="array",
            axes="SX",
            patch_size=[64],
            batch_size=2,
            num_epochs=1,
        )

        careamist = CAREamist(source=config, work_dir=tmp_path)
        careamist.train(train_source=sample_1d_data[:8])

        # Predict on single sample
        prediction = careamist.predict(source=sample_1d_data[8:9], data_type="array")

        assert prediction.shape == sample_1d_data[8:9].shape

    def test_n2v_1d_multichannel(self, tmp_path, sample_1d_multichannel_data):
        """Test N2V with multichannel 1D data."""
        config = create_n2v_configuration(
            experiment_name="test_1d_multichannel",
            data_type="array",
            axes="SCX",
            patch_size=[64],
            batch_size=2,
            num_epochs=2,
        )

        careamist = CAREamist(source=config, work_dir=tmp_path)
        careamist.train(train_source=sample_1d_multichannel_data)

        assert careamist.model is not None

    def test_n2v_1d_with_validation(self, tmp_path, sample_1d_data):
        """Test N2V training with validation split for 1D data."""
        config = create_n2v_configuration(
            experiment_name="test_1d_val",
            data_type="array",
            axes="SX",
            patch_size=[64],
            batch_size=2,
            num_epochs=2,
        )

        careamist = CAREamist(source=config, work_dir=tmp_path)

        # Train with validation split
        train_data = sample_1d_data[:8]
        val_data = sample_1d_data[8:]

        careamist.train(train_source=train_data, val_source=val_data)

        assert careamist.model is not None

    @pytest.mark.parametrize("axes", ["X", "SX", "TX", "CX"])
    def test_n2v_1d_different_axes(self, tmp_path, sample_1d_data, axes: str):
        """Test N2V with different 1D axes configurations."""
        # Prepare data based on axes
        if "S" not in axes and "T" not in axes and "C" not in axes:
            # Single sample
            data = sample_1d_data[0:1, 0, :]  # (1, L)
        elif "C" in axes:
            data = sample_1d_data[:, :, :]  # (S, C, L)
        else:
            data = sample_1d_data[:, 0, :]  # (S, L)

        config = create_n2v_configuration(
            experiment_name=f"test_1d_{axes}",
            data_type="array",
            axes=axes,
            patch_size=[32],
            batch_size=2,
            num_epochs=1,
        )

        careamist = CAREamist(source=config, work_dir=tmp_path)
        careamist.train(train_source=data)

        assert careamist.model is not None

    def test_n2v_1d_n2v2_mode(self, tmp_path, sample_1d_data):
        """Test N2V2 (with blur pool) for 1D data."""
        config = create_n2v_configuration(
            experiment_name="test_1d_n2v2",
            data_type="array",
            axes="SX",
            patch_size=[64],
            batch_size=2,
            num_epochs=1,
            use_n2v2=True,
        )

        careamist = CAREamist(source=config, work_dir=tmp_path)
        careamist.train(train_source=sample_1d_data)

        assert careamist.model is not None

    def test_n2v_1d_save_and_load(self, tmp_path, sample_1d_data):
        """Test saving and loading 1D N2V model."""
        config = create_n2v_configuration(
            experiment_name="test_1d_save_load",
            data_type="array",
            axes="SX",
            patch_size=[64],
            batch_size=2,
            num_epochs=1,
        )

        # Train and save
        careamist = CAREamist(source=config, work_dir=tmp_path)
        careamist.train(train_source=sample_1d_data)

        model_path = tmp_path / "model_1d.pth"
        careamist.save(model_path)

        # Load
        careamist_loaded = CAREamist(source=model_path)

        # Predict with loaded model
        prediction = careamist_loaded.predict(
            source=sample_1d_data[0:1], data_type="array"
        )

        assert prediction.shape == sample_1d_data[0:1].shape

    @pytest.mark.parametrize("patch_size", [32, 64, 128])
    def test_n2v_1d_different_patch_sizes(
        self, tmp_path, sample_1d_data, patch_size: int
    ):
        """Test N2V with different patch sizes for 1D data."""
        config = create_n2v_configuration(
            experiment_name=f"test_1d_patch_{patch_size}",
            data_type="array",
            axes="SX",
            patch_size=[patch_size],
            batch_size=2,
            num_epochs=1,
        )

        careamist = CAREamist(source=config, work_dir=tmp_path)
        careamist.train(train_source=sample_1d_data)

        assert careamist.model is not None

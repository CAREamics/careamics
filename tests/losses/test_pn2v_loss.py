"""Test PN2V loss function."""

from pathlib import Path

import numpy as np
import torch

from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.losses.fcn.losses import pn2v_loss
from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


def test_pn2v_loss_basic(tmp_path: Path, create_dummy_noise_model):
    """Test that PN2V loss function works with basic inputs."""
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    batch_size, height, width = 2, 32, 32

    samples = torch.randn(batch_size, 1, height, width)
    labels = torch.randn(batch_size, 1, height, width)
    masks = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    # Create a noise model config
    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel", path=tmp_path / "dummy_noise_model.npz"
    )

    # TODO Create noise model (we'll need to mock this properly)
    # For now, create a simple mock
    noise_model = GaussianMixtureNoiseModel(nm_config)

    # Test that loss function can be called
    try:
        loss_value = pn2v_loss(samples, labels, masks, noise_model)
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0  # Should be a scalar
        print(f"PN2V loss computed successfully: {loss_value.item()}")
    except Exception as e:
        print(f"Error calling pn2v_loss: {e}")
        raise

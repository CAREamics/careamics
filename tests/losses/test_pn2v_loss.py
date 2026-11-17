"""Test PN2V loss function."""

import torch

from careamics.losses import pn2v_loss
from careamics.models.lvae.noise_models import MultiChannelNoiseModel


def test_pn2v_loss_basic(pn2v_noise_model: MultiChannelNoiseModel):
    """The PN2V loss is a finite scalar."""
    batch_size, height, width = 2, 32, 32
    generator = torch.Generator().manual_seed(0)

    samples = torch.randn(batch_size, 4, height, width, generator=generator)
    labels = torch.randn(batch_size, 1, height, width, generator=generator)
    masks = torch.randint(0, 2, (batch_size, 1, height, width), generator=generator)

    loss_value = pn2v_loss(samples, labels, masks.float(), pn2v_noise_model.nmodel_0)

    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.dim() == 0
    assert torch.isfinite(loss_value)

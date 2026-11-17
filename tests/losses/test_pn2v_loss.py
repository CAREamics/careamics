"""Test PN2V loss function."""

import math

import numpy as np
import pytest
import torch

from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.losses import pn2v_loss
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)


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


def _random_gmm(seed: int) -> GaussianMixtureNoiseModel:
    """Return a 3-Gaussian noise model with fixed, non-trivial weights."""
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.1, (9, 2)).astype(np.float32)
    weight[3:6, 0] = np.log(25.0)
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


def _reference_single_channel_loss(samples, labels, masks, noise_model):
    """PN2V loss for one channel, as implemented before multichannel support."""
    likelihoods = noise_model.likelihood(labels, samples)
    likelihoods_avg = torch.log(torch.mean(likelihoods, dim=1, keepdim=True))
    return -torch.sum(likelihoods_avg * masks) / torch.sum(masks)


def test_pn2v_loss_single_channel_matches_reference():
    """With one channel the loss is bitwise identical to the original formula."""
    generator = torch.Generator().manual_seed(1)
    samples = torch.rand(2, 16, 8, 8, generator=generator) * 100
    labels = torch.rand(2, 1, 8, 8, generator=generator) * 100
    masks = torch.randint(0, 2, (2, 1, 8, 8), generator=generator).float()
    gmm = _random_gmm(0)
    multichannel = MultiChannelNoiseModel([gmm])

    expected = _reference_single_channel_loss(samples, labels, masks, gmm)

    assert torch.equal(pn2v_loss(samples, labels, masks, gmm), expected)
    assert torch.equal(pn2v_loss(samples, labels, masks, multichannel), expected)


def test_pn2v_loss_multichannel_uses_channel_noise_models():
    """Channel `c` owns samples `[c * K, (c + 1) * K)` and its own noise model."""
    generator = torch.Generator().manual_seed(2)
    n_samples = 5
    samples = torch.rand(2, 2 * n_samples, 8, 8, generator=generator) * 100
    labels = torch.rand(2, 2, 8, 8, generator=generator) * 100
    masks = torch.randint(0, 2, (2, 2, 8, 8), generator=generator).float()
    gmms = [_random_gmm(0), _random_gmm(1)]

    loss = pn2v_loss(samples, labels, masks, MultiChannelNoiseModel(gmms))

    per_channel_sums = [
        _reference_single_channel_loss(
            samples[:, c * n_samples : (c + 1) * n_samples],
            labels[:, c : c + 1],
            masks[:, c : c + 1],
            gmms[c],
        )
        * masks[:, c].sum()
        for c in range(2)
    ]
    expected = sum(per_channel_sums) / masks.sum()
    torch.testing.assert_close(loss, expected)


def test_pn2v_loss_rejects_mismatched_noise_model():
    """The noise model needs one channel per observation channel."""
    samples = torch.rand(1, 4, 4, 4)
    labels = torch.rand(1, 2, 4, 4)
    masks = torch.ones_like(labels)

    with pytest.raises(ValueError, match="covers one channel"):
        pn2v_loss(samples, labels, masks, _random_gmm(0))
    with pytest.raises(ValueError, match="channel"):
        pn2v_loss(samples, labels, masks, MultiChannelNoiseModel([_random_gmm(0)]))
    with pytest.raises(ValueError, match="multiple"):
        pn2v_loss(torch.rand(1, 3, 4, 4), labels, masks, _random_gmm(0))


def test_pn2v_loss_known_value(pn2v_noise_model: MultiChannelNoiseModel):
    """With `N(obs; s, 100)`, two samples at distances 0 and 10 from the observation
    give `-log(p0 * (1 + exp(-1/2)) / 2)`, with `p0 = 1 / sqrt(2 pi 100)`."""
    samples = torch.tensor([0.0, 10.0]).view(1, 2, 1, 1)
    observations = torch.zeros(1, 1, 1, 1)
    masks = torch.ones_like(observations)

    loss = pn2v_loss(samples, observations, masks, pn2v_noise_model)

    p0 = 1 / math.sqrt(2 * math.pi * 100)
    expected = -math.log(p0 * (1 + math.exp(-0.5)) / 2)
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert loss.item() == pytest.approx(3.4406, abs=1e-4)


def test_pn2v_loss_ignores_unmasked_pixels(pn2v_noise_model: MultiChannelNoiseModel):
    """Only masked pixels contribute: changing unmasked pixels leaves the loss as is."""
    generator = torch.Generator().manual_seed(3)
    samples = torch.rand(2, 4, 8, 8, generator=generator) * 100
    observations = torch.rand(2, 1, 8, 8, generator=generator) * 100
    masks = torch.zeros_like(observations)
    masks[:, :, ::3, ::3] = 1
    unmasked = masks.expand_as(samples) == 0

    loss = pn2v_loss(samples, observations, masks, pn2v_noise_model)
    garbage_samples = torch.where(unmasked, torch.full_like(samples, 5000), samples)
    garbage_observations = torch.where(
        masks == 0, torch.full_like(observations, -300), observations
    )
    garbage_loss = pn2v_loss(
        garbage_samples, garbage_observations, masks, pn2v_noise_model
    )

    assert torch.equal(loss, garbage_loss)

from types import SimpleNamespace

import lightning.pytorch as L
import pytest
import torch
from torch import nn

from careamics.lightning.modules.module_utils import (
    compile_model_if_requested,
    mmse_and_sample_std,
    request_model_compilation,
    skip_nan_batch,
    zero_gradient_loss,
)


def test_mmse_and_sample_std_matches_the_drawn_samples() -> None:
    """The mean and standard deviation are those of the drawn samples."""
    samples = [torch.full((2, 3, 4, 4), float(i)) for i in range(4)]
    it = iter(samples)

    mean, std = mmse_and_sample_std(lambda _: next(it), torch.empty(0), n_samples=4)

    expected = torch.stack(samples, dim=0)
    torch.testing.assert_close(mean, expected.mean(dim=0))
    torch.testing.assert_close(std, expected.std(dim=0))


def test_mmse_and_sample_std_draws_n_samples() -> None:
    """`sample_prediction` is called exactly `n_samples` times."""
    calls = 0

    def sample(x_data: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.randn(1, 1, 8, 8)

    mmse_and_sample_std(sample, torch.empty(0), n_samples=7)
    assert calls == 7


def test_mmse_and_sample_std_undefined_for_a_single_sample() -> None:
    """With one sample the standard deviation is undefined and reported as None."""
    sample = torch.randn(2, 3, 4, 4)

    mean, std = mmse_and_sample_std(lambda _: sample, torch.empty(0), n_samples=1)

    torch.testing.assert_close(mean, sample)
    assert std is None


def test_mmse_and_sample_std_is_exact_under_affine_sampling() -> None:
    """Sampling in denormalized space scales the std without offsetting it."""
    samples = [torch.randn(2, 1, 4, 4) for _ in range(5)]
    data_mean, data_std = 12.0, 3.0

    it = iter(samples)
    _, normalized_std = mmse_and_sample_std(
        lambda _: next(it), torch.empty(0), n_samples=5
    )

    it = iter(samples)
    _, denormalized_std = mmse_and_sample_std(
        lambda _: next(it) * data_std + data_mean, torch.empty(0), n_samples=5
    )

    torch.testing.assert_close(denormalized_std, normalized_std * data_std)


class _StubModule(L.LightningModule):
    """A minimal module exposing the `model` attribute the helpers operate on."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Conv2d(2, 1, 3))


def _attach_world_size(module: L.LightningModule, world_size: int) -> None:
    """Fake a trainer just far enough for the helpers to read `world_size`."""
    module._trainer = SimpleNamespace(world_size=world_size)


def test_zero_gradient_loss_touches_every_trainable_parameter() -> None:
    """Every parameter receives a gradient, and every gradient is zero."""
    model = _StubModule().model

    zero_gradient_loss(model).backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all((parameter.grad == 0).all() for parameter in model.parameters())


def test_zero_gradient_loss_ignores_frozen_parameters() -> None:
    """Frozen parameters are skipped rather than making the loss fail."""
    model = _StubModule().model
    frozen, *_ = model.parameters()
    frozen.requires_grad_(False)

    zero_gradient_loss(model).backward()

    assert frozen.grad is None


def test_zero_gradient_loss_rejects_a_model_without_parameters() -> None:
    """A model with nothing to train cannot produce a usable loss."""
    with pytest.raises(ValueError, match="no trainable parameters"):
        zero_gradient_loss(nn.Identity())


@pytest.mark.parametrize("world_size", [1, 2])
def test_skip_nan_batch_neutralizes_the_step(world_size: int) -> None:
    """The batch is neutralized with a zero gradient, on one device and on many.

    Returning `None` -- Lightning's documented skip -- is not usable: its AMP
    plugin runs gradient clipping before it checks for a `None` closure result,
    so an un-populated `.grad` makes `clip_grad_value_` raise on the empty list.
    Under DDP a `None` would additionally desynchronize the all-reduce.
    """
    module = _StubModule()
    _attach_world_size(module, world_size=world_size)

    with pytest.warns(UserWarning, match="NaN loss"):
        loss = skip_nan_batch(module)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    loss.backward()
    assert all((parameter.grad == 0).all() for parameter in module.model.parameters())


def test_skip_nan_batch_populates_every_gradient() -> None:
    """Clipping needs a `.grad` on every parameter, not merely a zero update."""
    module = _StubModule()
    _attach_world_size(module, world_size=1)

    with pytest.warns(UserWarning, match="NaN loss"):
        skip_nan_batch(module).backward()

    assert all(p.grad is not None for p in module.model.parameters())


def test_compile_model_if_requested_does_nothing_unrequested() -> None:
    """Without a request the model is left untouched."""
    module = _StubModule()

    compile_model_if_requested(module)

    assert module.model._compiled_call_impl is None


def test_compile_model_if_requested_preserves_state_dict_keys() -> None:
    """Compiling in place must not rename any checkpoint key.

    Rebinding `model` to the object returned by `torch.compile` would prefix
    every key with `_orig_mod.` and break checkpoint interchange with
    uncompiled runs.
    """
    module = _StubModule()
    before = list(module.state_dict().keys())

    request_model_compilation(module)
    compile_model_if_requested(module)

    assert module.model._compiled_call_impl is not None
    assert list(module.state_dict().keys()) == before


def test_compile_model_if_requested_is_idempotent() -> None:
    """`configure_model` runs once per Trainer entry point, so repeats are no-ops."""
    module = _StubModule()
    request_model_compilation(module)

    compile_model_if_requested(module)
    compiled = module.model._compiled_call_impl
    compile_model_if_requested(module)

    assert module.model._compiled_call_impl is compiled

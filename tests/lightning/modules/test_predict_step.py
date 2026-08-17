"""Tests for the uncertainty output shared by the sampling-based modules."""

from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from careamics.dataset.image_region_data import ImageRegionData
from careamics.lightning.modules.hdn_module import HDNModule
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.lightning.prediction import decollate_image_region_data

SAMPLING_MODULES = [HDNModule, MicroSplitModule]


class _NoisyModel:
    """Stand-in for the LVAE, drawing a different reconstruction on every call."""

    def reset_for_inference(self, spatial_shape: Any) -> None:
        pass

    def __call__(self, x_data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return x_data + torch.randn_like(x_data), {}


class _IdentityNormalization:
    """Stand-in for a normalization leaving the reconstruction untouched."""

    def denormalize(self, patch: torch.Tensor) -> torch.Tensor:
        return patch


def _module_stub(module_cls, n_samples: int, output_channels: int = 1):
    """Provide the module state read by `predict_step`, and its bound methods."""
    stub = SimpleNamespace(
        model=_NoisyModel(),
        n_samples=n_samples,
        _get_reconstruction=lambda model_outputs: model_outputs[0],
        _trainer=SimpleNamespace(
            datamodule=SimpleNamespace(
                predict_dataset=SimpleNamespace(normalization=_IdentityNormalization())
            )
        ),
        config=SimpleNamespace(model=SimpleNamespace(output_channels=output_channels)),
    )
    stub.predict_sample = partial(module_cls.predict_sample, stub)
    return stub


def _input_region(channels: int = 1, batch_size: int = 2) -> ImageRegionData:
    """Provide a collated input batch of 8x8 uint16 images.

    Collation gives every per-item field a length `batch_size` entry, which is what
    `decollate_image_region_data` indexes back apart.
    """
    per_item = lambda value: torch.full((batch_size,), value)  # noqa: E731
    return ImageRegionData(
        data=torch.rand(batch_size, channels, 8, 8),
        source=["array"] * batch_size,
        data_shape=[per_item(batch_size), per_item(channels), per_item(8), per_item(8)],
        dtype=["uint16"] * batch_size,
        axes=["YX"] * batch_size,
        original_data_shape=[per_item(batch_size), per_item(8), per_item(8)],
        region_spec={
            "data_idx": torch.zeros(batch_size, dtype=torch.int64),
            "sample_idx": torch.arange(batch_size),
        },
        additional_metadata={},
    )


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
def test_predict_step_has_no_uncertainty_for_a_single_sample(module_cls):
    """A single draw leaves the uncertainty undefined rather than reporting zeros."""
    stub = _module_stub(module_cls, n_samples=1)

    prediction, uncertainty = module_cls.predict_step(stub, (_input_region(),), 0)

    assert isinstance(prediction, ImageRegionData)
    assert uncertainty is None


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
def test_predict_step_returns_uncertainty_for_several_samples(module_cls):
    """The uncertainty shares the region metadata of the prediction."""
    stub = _module_stub(module_cls, n_samples=3)

    prediction, uncertainty = module_cls.predict_step(stub, (_input_region(),), 0)

    assert uncertainty is not None
    assert uncertainty.data.shape == prediction.data.shape
    assert uncertainty.region_spec == prediction.region_spec
    assert uncertainty.source == prediction.source
    assert np.all(uncertainty.data > 0)


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
@pytest.mark.parametrize("batch_size", [2, 8])
def test_predict_step_uncertainty_decollates(module_cls, batch_size):
    """The uncertainty carries per-item metadata, so it survives decollation."""
    stub = _module_stub(module_cls, n_samples=3)
    batch = _input_region(batch_size=batch_size)

    _, uncertainty = module_cls.predict_step(stub, (batch,), 0)
    regions = decollate_image_region_data(uncertainty)

    assert len(regions) == batch_size
    assert all(region.data.shape == (1, 8, 8) for region in regions)


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
def test_predict_step_data_shape_follows_the_output_channels(module_cls):
    """`data_shape` describes the prediction, not the multi-channel input."""
    stub = _module_stub(module_cls, n_samples=3, output_channels=1)

    batch = _input_region(channels=3, batch_size=2)

    prediction, uncertainty = module_cls.predict_step(stub, (batch,), 0)

    assert prediction.data_shape[1].tolist() == [1, 1]
    assert uncertainty.data_shape[1].tolist() == [1, 1]

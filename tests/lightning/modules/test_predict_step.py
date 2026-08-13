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


def _input_region(channels: int = 1) -> ImageRegionData:
    """Provide a collated input batch of two 8x8 uint16 images."""
    return ImageRegionData(
        data=torch.rand(2, channels, 8, 8),
        source=["array", "array"],
        data_shape=[torch.tensor([2]), torch.tensor([channels])]
        + [torch.tensor([8])] * 2,
        dtype=["uint16", "uint16"],
        axes=["YX", "YX"],
        original_data_shape=[torch.tensor([2])] + [torch.tensor([8])] * 2,
        region_spec={
            "data_idx": torch.tensor([0, 0]),
            "sample_idx": torch.tensor([0, 1]),
        },
        additional_metadata=[{}, {}],
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
def test_predict_step_uncertainty_is_float(module_cls):
    """The uncertainty is a float quantity, whatever the dtype of the input."""
    stub = _module_stub(module_cls, n_samples=3)

    prediction, uncertainty = module_cls.predict_step(stub, (_input_region(),), 0)

    assert prediction.dtype == ["uint16", "uint16"]
    assert uncertainty.dtype == "float32"


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
def test_predict_step_data_shape_follows_the_output_channels(module_cls):
    """`data_shape` describes the prediction, not the multi-channel input."""
    stub = _module_stub(module_cls, n_samples=3, output_channels=1)

    prediction, uncertainty = module_cls.predict_step(stub, (_input_region(3),), 0)

    assert prediction.data_shape[1] == torch.tensor([1])
    assert uncertainty.data_shape[1] == torch.tensor([1])

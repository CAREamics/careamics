"""Tests for the prediction regions returned by `predict_step`."""

from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from careamics.config.data import DataConfig
from careamics.dataset.dataset import CareamicsDataset
from careamics.dataset.factory import create_dataset
from careamics.lightning.modules.hdn_module import HDNModule
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.lightning.modules.seg_unet_module import SegModule

SAMPLING_MODULES = [HDNModule, MicroSplitModule]


class _NoisyModel:
    """Stand-in for the LVAE, drawing a different reconstruction on every call."""

    def __init__(self, output_channels: int) -> None:
        self.output_channels = output_channels

    def reset_for_inference(self, spatial_shape: Any) -> None:
        pass

    def __call__(self, x_data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        reconstruction = x_data.mean(dim=1, keepdim=True)
        reconstruction = reconstruction.repeat(1, self.output_channels, 1, 1)
        return reconstruction + torch.randn_like(reconstruction), {}


def _predict_dataset(axes: str, n_channels: int) -> CareamicsDataset:
    shape = (4, n_channels, 8, 8) if "C" in axes else (4, 8, 8)
    data = np.random.default_rng(42).random(shape).astype(np.float32)
    config = DataConfig(
        mode="predicting",
        data_type="array",
        axes=axes,
        patching={"name": "whole"},
        normalization={"name": "none"},
        batch_size=1,
    )
    return create_dataset(config=config, inputs=[data], targets=None)


def _batch(dataset: CareamicsDataset) -> tuple[torch.Tensor, ...]:
    return default_collate([dataset[i] for i in range(len(dataset))])


def _sampling_stub(
    module_cls, dataset: CareamicsDataset, n_samples: int, output_channels: int = 1
):
    stub = SimpleNamespace(
        model=_NoisyModel(output_channels),
        n_samples=n_samples,
        _get_reconstruction=lambda model_outputs: model_outputs[0],
        _trainer=SimpleNamespace(datamodule=SimpleNamespace(predict_dataset=dataset)),
    )
    stub.predict_sample = partial(module_cls.predict_sample, stub)
    return stub


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
def test_predict_step_has_no_uncertainty_for_a_single_sample(module_cls):
    """A single draw leaves the uncertainty undefined rather than reporting zeros."""
    dataset = _predict_dataset("SYX", 1)
    stub = _sampling_stub(module_cls, dataset, n_samples=1)

    prediction, uncertainty = module_cls.predict_step(stub, _batch(dataset), 0)

    assert len(prediction) == len(dataset)
    assert uncertainty is None


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
def test_predict_step_returns_uncertainty_for_several_samples(module_cls):
    """Each uncertainty region shares the metadata of its prediction region."""
    dataset = _predict_dataset("SYX", 1)
    stub = _sampling_stub(module_cls, dataset, n_samples=3)

    prediction, uncertainty = module_cls.predict_step(stub, _batch(dataset), 0)

    assert uncertainty is not None
    assert len(uncertainty) == len(prediction) == len(dataset)
    for pred_region, unc_region in zip(prediction, uncertainty, strict=True):
        assert unc_region.data.shape == pred_region.data.shape == (1, 8, 8)
        assert unc_region.region_spec == pred_region.region_spec
        assert unc_region.source == pred_region.source
        assert np.all(unc_region.data > 0)
    assert [region.region_spec["sample_idx"] for region in prediction] == [0, 1, 2, 3]


@pytest.mark.parametrize("module_cls", SAMPLING_MODULES)
@pytest.mark.parametrize(
    "axes, input_channels, output_channels, expected_target_axes",
    [
        ("SCYX", 3, 1, "SCYX"),
        ("SYX", 1, 2, "SYXC"),
    ],
)
def test_predict_step_metadata_follows_the_output_channels(
    module_cls, axes, input_channels, output_channels, expected_target_axes
):
    """`data_shape` and `target_axes` describe the prediction, not the input."""
    dataset = _predict_dataset(axes, input_channels)
    stub = _sampling_stub(
        module_cls, dataset, n_samples=3, output_channels=output_channels
    )

    prediction, uncertainty = module_cls.predict_step(stub, _batch(dataset), 0)

    assert uncertainty is not None
    for region in [*prediction, *uncertainty]:
        assert region.data_shape[1] == output_channels
        assert region.target_axes == expected_target_axes


@pytest.mark.parametrize(
    "prediction_output, n_channels, expected_target_axes",
    [("probabilities", 3, "SYXC"), ("labels", 1, "SYX")],
)
def test_segmentation_prediction_keeps_the_class_axis(
    prediction_output, n_channels, expected_target_axes
):
    """Probabilities keep one channel per class, labels have no channel axis."""
    dataset = _predict_dataset("SYX", 1)
    stub = SimpleNamespace(
        model=lambda x: x.repeat(1, 3, 1, 1),
        config=SimpleNamespace(prediction_output=prediction_output),
        _trainer=SimpleNamespace(datamodule=SimpleNamespace(predict_dataset=dataset)),
    )

    prediction = SegModule.predict_step(stub, _batch(dataset), 0)

    assert len(prediction) == len(dataset)
    for region in prediction:
        assert region.data.shape == (n_channels, 8, 8)
        assert region.target_axes == expected_target_axes

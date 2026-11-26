"""Tests for segmentation prediction with test-time augmentation."""

from types import SimpleNamespace

import numpy as np
import torch

from careamics.dataset.image_region_data import ImageRegionData
from careamics.lightning.modules.seg_unet_module import SegModule
from careamics.lightning.prediction.tta import TTAVariant

_IDENTITY_LOGITS = torch.tensor([-5.0, 4.0, 4.0])
_FLIPPED_LOGITS = torch.tensor([4.0, 1.0, -5.0])


def _orientation_logits(tensor: torch.Tensor) -> torch.Tensor:
    """Return class logits that depend on whether X has been flipped."""
    flipped = bool((tensor[0, 0, 0, 0] < tensor[0, 0, 0, -1]).item())
    logits = _FLIPPED_LOGITS if flipped else _IDENTITY_LOGITS
    return (
        logits.view(1, -1, 1, 1)
        .expand(tensor.shape[0], -1, tensor.shape[-2], tensor.shape[-1])
        .contiguous()
    )


def test_predict_step_tta_mask_is_argmax_of_mean_probabilities():
    """Test that the TTA segmentation mask is the argmax of the mean probabilities."""
    image = torch.tensor([[[[1.0, 0.0]]]])
    flipped = torch.flip(image, dims=(-1,))
    mean_probabilities = (
        _orientation_logits(image).softmax(dim=1)
        + torch.flip(_orientation_logits(flipped).softmax(dim=1), dims=(-1,))
    ) / 2
    expected = mean_probabilities.argmax(dim=1, keepdim=True).numpy()
    mean_logits = (_orientation_logits(image) + _orientation_logits(flipped)) / 2
    assert not np.array_equal(expected, mean_logits.argmax(dim=1, keepdim=True).numpy())

    region = ImageRegionData(
        data=image,
        source="array",
        data_shape=(1, 1, 1, 2),
        dtype="float32",
        axes="SYX",
        target_axes="SYX",
        original_data_shape=(1, 1, 2),
        region_spec={
            "data_idx": 0,
            "sample_idx": 0,
            "coords": (0, 0),
            "patch_size": (1, 2),
        },
        additional_metadata={},
    )
    module = SimpleNamespace(
        model=_orientation_logits,
        tta_variants=[TTAVariant(0, None), TTAVariant(0, -1)],
        config=SimpleNamespace(prediction_output="mask"),
    )

    output = SegModule.predict_step(module, (region,), 0)

    np.testing.assert_array_equal(output.data, expected)

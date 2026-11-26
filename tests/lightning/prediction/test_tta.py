import pytest
import torch

from careamics.config.augmentations import (
    SwapInputTargetConfig,
    XYFlipConfig,
    XYRandomRotate90Config,
)
from careamics.lightning.prediction.tta import (
    _transform,
    get_tta_variants,
    predict_with_tta,
)


@pytest.mark.parametrize(
    "augmentations, n_variants",
    [
        ([XYFlipConfig(flip_x=True, flip_y=False)], 2),
        ([XYFlipConfig(flip_x=True, flip_y=True)], 3),
        ([XYRandomRotate90Config()], 4),
        ([XYFlipConfig(), XYRandomRotate90Config()], 8),
        ([XYFlipConfig(p=0), XYRandomRotate90Config(p=0)], 1),
        ([SwapInputTargetConfig()], 1),
        ([], 1),
    ],
)
def test_get_tta_variants(augmentations, n_variants):
    """Test the number of unique TTA variants for each augmentation set."""
    variants = get_tta_variants(augmentations)
    assert len(set(variants)) == n_variants


@pytest.mark.parametrize("shape", [(2, 1, 8, 12), (2, 1, 4, 8, 12)])
def test_transform_inverse(shape):
    """Test that the inverse transform restores the input."""
    x = torch.randn(shape)
    for variant in get_tta_variants([XYFlipConfig(), XYRandomRotate90Config()]):
        torch.testing.assert_close(_transform(_transform(x, variant), variant, True), x)


def test_predict_with_tta():
    """Test that TTA is a no-op for an equivariant model and not otherwise."""
    x = torch.randn(1, 1, 8, 8)
    variants = get_tta_variants([XYFlipConfig(), XYRandomRotate90Config()])

    torch.testing.assert_close(predict_with_tta(lambda t: 2 * t, x, variants), 2 * x)

    def model(t: torch.Tensor) -> torch.Tensor:
        return t + torch.arange(t.shape[-1], dtype=t.dtype)

    assert not torch.allclose(predict_with_tta(model, x, variants), model(x))

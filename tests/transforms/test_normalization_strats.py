import numpy as np
import pytest

from careamics.dataset.dataset_utils.running_stats import compute_normalization_stats
from careamics.transforms.normalization import NoNormalization, QuantileNormalization, MeanStdNormalization
from careamics.transforms.normalization.mean_std_normalization import _reshape_stats

@pytest.mark.parametrize(
    "patch_shape,axes",
    [
        ((1, 1, 10, 10), "YX"),    
        ((1, 3, 10, 10), "CYX"),
        ((1, 3, 5, 10, 10), "CZYX"), 
    ],
)
def test_mean_std_normalization(patch_shape, axes):
    patch = np.random.randint(0, 65535, size=patch_shape, dtype=np.uint16).astype(np.float32)
    means, stds = compute_normalization_stats(image=patch)
    mean_std_norm = MeanStdNormalization(image_means=means, image_stds=stds)
    normalized_patch, *_ = mean_std_norm(patch[0])
    assert np.abs(normalized_patch.mean()) < 0.02
    assert np.abs(normalized_patch.std() - 1) < 0.2


@pytest.mark.parametrize(
    "patch_shape,axes",
    [
        ((1, 1, 10, 10), "YX"),    
        ((1, 3, 10, 10), "CYX"),
        ((1, 3, 5, 10, 10), "CZYX"), 
    ],
)
def test_mean_std_denormalization(patch_shape, axes):
    patch = np.random.randint(0, 65535, size=patch_shape, dtype=np.uint16).astype(np.float32)
    means, stds = compute_normalization_stats(image=patch)
    mean_std_norm = MeanStdNormalization(image_means=means, image_stds=stds)
    normalized_patch, *_ = mean_std_norm(patch=patch[0])
    denormalized_patch = mean_std_norm.denormalize(patch=normalized_patch[None,])
    assert np.isclose(denormalized_patch, patch, atol=1).all()


@pytest.mark.parametrize(
    "patch_shape,axes",
    [
        ((1, 1, 10, 10), "YX"),    
        ((1, 3, 10, 10), "CYX"),
        ((1, 3, 5, 10, 10), "CZYX"), 
    ],
)
def test_no_normalization(patch_shape, axes):
    patch = np.random.randint(0, 65535, size=patch_shape, dtype=np.uint16).astype(np.float32)
    no_norm = NoNormalization()
    normalized_patch, *_ = no_norm(patch=patch[0])
    assert np.isclose(normalized_patch, patch, atol=1).all()
    denormalized_patch = no_norm.denormalize(patch=normalized_patch[None,])
    assert np.isclose(denormalized_patch, patch, atol=1).all()
    

@pytest.mark.parametrize(
    "patch_shape,axes",
    [
        ((1, 1, 10, 10), "YX"),    
        ((1, 3, 10, 10), "CYX"),
        ((1, 3, 5, 10, 10), "CZYX"), 
    ],
)
def test_quantile_normalization(patch_shape, axes):
    patch = np.random.randint(0, 65535, size=patch_shape, dtype=np.uint16).astype(np.float32)
    quantile_norm = QuantileNormalization()
    normalized_patch, *_ = quantile_norm(patch=patch[0])
    assert np.allclose(normalized_patch.min(), 0, atol=0.05)
    assert np.allclose(normalized_patch.max(), 1, atol=0.05)

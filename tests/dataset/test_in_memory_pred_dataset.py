import numpy as np
import pytest

from careamics.config import InferenceConfig
from careamics.dataset import InMemoryPredDataset


@pytest.mark.parametrize(
    "shape, axes, expected_shape",
    [
        ((16, 16), "YX", (1, 16, 16)),
        ((3, 16, 16), "CYX", (3, 16, 16)),
        ((8, 16, 16), "ZYX", (1, 8, 16, 16)),
        ((3, 8, 16, 16), "CZYX", (3, 8, 16, 16)),
        ((4, 16, 16), "SYX", (1, 16, 16)),
        ((4, 3, 16, 16), "SCYX", (3, 16, 16)),
        ((4, 3, 8, 16, 16), "SCZYX", (3, 8, 16, 16)),
    ],
)
def test_correct_normalized_outputs(shape, axes, expected_shape):
    """Test that the dataset returns normalized images with singleton
    sample dimension."""
    rng = np.random.default_rng(42)

    # check expected length
    if "S" in axes:
        # find index of S and check shape
        idx = axes.index("S")
        n_patches = shape[idx]
    else:
        n_patches = 1

    # check number of channels
    if "C" in axes:
        # get index
        idx = axes.index("C")
        n_channels = shape[idx]
    else:
        n_channels = 1

    # create array
    array = 255 * rng.random(shape)

    # create config
    config = InferenceConfig(
        data_type="array",
        axes=axes,
        image_means=[np.mean(array)] * n_channels,
        image_stds=[np.std(array)] * n_channels,
    )

    # create dataset
    dataset = InMemoryPredDataset(config, array)

    # check length
    assert len(dataset) == n_patches

    # check that the dataset returns normalized images
    for i in range(len(dataset)):
        img = dataset[i][0]

        # check that it has the correct shape
        assert img.shape == expected_shape

        # check that the image is normalized
        assert np.isclose(np.mean(img), 0, atol=0.1)
        assert np.isclose(np.std(img), 1, atol=0.1)

        # check that they are independent slices
        for j in range(i + 1, len(dataset)):
            img2 = dataset[j]
            assert not np.allclose(img, img2)

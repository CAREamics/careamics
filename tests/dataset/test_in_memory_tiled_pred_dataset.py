import numpy as np
import pytest

from careamics.config import InferenceConfig
from careamics.dataset import InMemoryTiledPredDataset


# TODO extract tiles is returning C(Z)YX and no singleton S!
@pytest.mark.parametrize(
    "shape, axes, expected_shape",
    [
        ((16, 16), "YX", (1, 16, 16)),
        ((3, 16, 16), "CYX", (3, 16, 16)),
        ((16, 16, 16), "ZYX", (1, 16, 16, 16)),
        ((3, 16, 16, 16), "CZYX", (3, 16, 16, 16)),
        ((4, 16, 16), "SYX", (1, 16, 16)),
        ((4, 3, 16, 16), "SCYX", (3, 16, 16)),
        ((4, 3, 16, 16, 16), "SCZYX", (3, 16, 16, 16)),
    ],
)
def test_correct_normalized_outputs(shape, axes, expected_shape):
    """Test that the dataset returns normalized images with singleton
    sample dimension."""
    rng = np.random.default_rng(42)

    tile_size = (8, 8, 8) if "Z" in axes else (8, 8)
    tile_overlap = (4, 4, 4) if "Z" in axes else (4, 4)

    # check expected length
    n_tiles = np.prod(
        np.ceil(
            (expected_shape[1:] - np.array(tile_overlap))
            / (np.array(tile_size) - np.array(tile_overlap))
        )
    ).astype(int)

    # check number of samples
    if "S" in axes:
        # get index
        idx = axes.index("S")
        n_samples = shape[idx]
    else:
        n_samples = 1

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
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    # create dataset
    dataset = InMemoryTiledPredDataset(config, array)

    # check length
    assert len(dataset) == n_samples * n_tiles

    # check that the dataset returns normalized images
    for i in range(len(dataset)):
        img, _ = dataset[i]

        # check that it has the correct shape
        assert img.shape == (n_channels,) + tile_size

        # check that the image is normalized
        assert np.isclose(np.mean(img), 0, atol=0.25)
        assert np.isclose(np.std(img), 1, atol=0.2)

        # check that they are independent slices
        for j in range(i + 1, len(dataset)):
            img2, _ = dataset[j]
            assert not np.allclose(img, img2)

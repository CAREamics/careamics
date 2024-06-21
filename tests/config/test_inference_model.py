import pytest

from careamics.config.inference_model import InferenceConfig


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_inference: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_inference["data_type"] = ext

    # instantiate InferenceModel model
    with pytest.raises(ValueError):
        InferenceConfig(**minimum_inference)


def test_mean_std_both_specified_or_none(minimum_inference: dict):
    """Test error raising when setting mean and std."""
    # Errors if both are None
    minimum_inference["image_means"] = []
    minimum_inference["image_stds"] = []
    with pytest.raises(ValueError):
        InferenceConfig(**minimum_inference)

    # Error if only mean is defined
    minimum_inference["image_means"] = [10.4]
    with pytest.raises(ValueError):
        InferenceConfig(**minimum_inference)

    # Error if only std is defined
    minimum_inference.pop("image_means")
    minimum_inference["image_stds"] = [10.4]
    with pytest.raises(ValueError):
        InferenceConfig(**minimum_inference)

    # No error if both are specified
    minimum_inference["image_means"] = [10.4]
    minimum_inference["image_stds"] = [10.4]
    InferenceConfig(**minimum_inference)


def test_tile_size(minimum_inference: dict):
    """Test that non-zero even patch size are accepted."""
    # no tiling
    prediction_model = InferenceConfig(**minimum_inference)

    # 2D
    minimum_inference["tile_size"] = [16, 8]
    minimum_inference["tile_overlap"] = [2, 2]
    minimum_inference["axes"] = "YX"

    prediction_model = InferenceConfig(**minimum_inference)
    assert prediction_model.tile_size == minimum_inference["tile_size"]
    assert prediction_model.tile_overlap == minimum_inference["tile_overlap"]

    # 3D
    minimum_inference["tile_size"] = [16, 8, 32]
    minimum_inference["tile_overlap"] = [2, 2, 2]
    minimum_inference["axes"] = "ZYX"

    prediction_model = InferenceConfig(**minimum_inference)
    assert prediction_model.tile_size == minimum_inference["tile_size"]
    assert prediction_model.tile_overlap == minimum_inference["tile_overlap"]


@pytest.mark.parametrize(
    "tile_size", [[12], [0, 12, 12], [12, 12, 13], [12, 12, 12, 12]]
)
def test_wrong_tile_size(minimum_inference: dict, tile_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_inference["axes"] = "ZYX" if len(tile_size) == 3 else "YX"
    minimum_inference["tile_size"] = tile_size

    with pytest.raises(ValueError):
        InferenceConfig(**minimum_inference)


@pytest.mark.parametrize(
    "tile_size, tile_overlap", [([12, 12], [2, 2, 2]), ([12, 12, 12], [14, 2, 2])]
)
def test_wrong_tile_overlap(minimum_inference: dict, tile_size, tile_overlap):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_inference["axes"] = "ZYX" if len(tile_size) == 3 else "YX"
    minimum_inference["tile_size"] = tile_size
    minimum_inference["tile_overlap"] = tile_overlap

    with pytest.raises(ValueError):
        InferenceConfig(**minimum_inference)


def test_set_3d(minimum_inference: dict):
    """Test that 3D can be set."""
    minimum_inference["tile_size"] = [64, 64]
    minimum_inference["tile_overlap"] = [32, 32]

    pred = InferenceConfig(**minimum_inference)
    assert "Z" not in pred.axes
    assert len(pred.tile_size) == 2
    assert len(pred.tile_overlap) == 2

    # error if changing Z manually
    with pytest.raises(ValueError):
        pred.axes = "ZYX"

    # or patch size
    pred = InferenceConfig(**minimum_inference)
    with pytest.raises(ValueError):
        pred.tile_size = [64, 64, 64]

    with pytest.raises(ValueError):
        pred.tile_overlap = [64, 64, 64]

    # set 3D
    pred = InferenceConfig(**minimum_inference)
    pred.set_3D("ZYX", [64, 64, 64], [32, 32, 32])
    assert "Z" in pred.axes
    assert len(pred.tile_size) == 3
    assert len(pred.tile_overlap) == 3

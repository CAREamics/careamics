import numpy as np
import pytest
import yaml

from careamics.config.data_model import DataConfig
from careamics.config.support import (
    SupportedTransform,
)
from careamics.config.transformations import (
    N2VManipulateModel,
    NormalizeModel,
    XYFlipModel,
    XYRandomRotate90Model,
)


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_algorithm_n2v: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_algorithm_n2v["data_type"] = ext

    # instantiate DataModel model
    with pytest.raises(ValueError):
        DataConfig(**minimum_algorithm_n2v)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(minimum_algorithm_n2v: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    minimum_algorithm_n2v["image_means"] = [mean]
    minimum_algorithm_n2v["image_stds"] = [std]
    minimum_algorithm_n2v["target_means"] = [mean]
    minimum_algorithm_n2v["target_stds"] = [std]

    data_model = DataConfig(**minimum_algorithm_n2v)
    assert data_model.image_means == [mean]
    assert data_model.image_stds == [std]
    assert data_model.target_means == [mean]
    assert data_model.target_stds == [std]


def test_mean_std_both_specified_or_none(minimum_algorithm_n2v: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    DataConfig(**minimum_algorithm_n2v)

    # Error if only mean is defined
    minimum_algorithm_n2v["image_means"] = [10.4]
    with pytest.raises(ValueError):
        DataConfig(**minimum_algorithm_n2v)

    # Error if only std is defined
    minimum_algorithm_n2v.pop("image_means")
    minimum_algorithm_n2v["image_stds"] = [10.4]
    with pytest.raises(ValueError):
        DataConfig(**minimum_algorithm_n2v)

    # No error if both are specified
    minimum_algorithm_n2v["image_means"] = [10.4]
    minimum_algorithm_n2v["image_stds"] = [10.4]
    DataConfig(**minimum_algorithm_n2v)

    # Error if target mean is defined but target std is None
    minimum_algorithm_n2v["target_stds"] = [10.4, 11]
    with pytest.raises(ValueError):
        DataConfig(**minimum_algorithm_n2v)


def test_set_mean_and_std(minimum_algorithm_n2v: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they None
    mean = [4.07]
    std = [14.07]
    data = DataConfig(**minimum_algorithm_n2v)
    data.set_means_and_stds(mean, std)
    assert data.image_means == mean
    assert data.image_stds == std

    # Set also target mean and std
    data.set_means_and_stds(mean, std, mean, std)
    assert data.target_means == mean
    assert data.target_stds == std


def test_normalize_not_accepted(minimum_algorithm_n2v: dict):
    """Test that normalize is not accepted, because it is mandatory and applied else
    where."""
    minimum_algorithm_n2v["image_means"] = [10.4]
    minimum_algorithm_n2v["image_stds"] = [3.2]
    minimum_algorithm_n2v["transforms"] = [
        NormalizeModel(image_means=[0.485], image_stds=[0.229])
    ]

    with pytest.raises(ValueError):
        DataConfig(**minimum_algorithm_n2v)


def test_patch_size(minimum_algorithm_n2v: dict):
    """Test that non-zero even patch size are accepted."""
    # 2D
    data_model = DataConfig(**minimum_algorithm_n2v)

    # 3D
    minimum_algorithm_n2v["patch_size"] = [16, 8, 8]
    minimum_algorithm_n2v["axes"] = "ZYX"

    data_model = DataConfig(**minimum_algorithm_n2v)
    assert data_model.patch_size == minimum_algorithm_n2v["patch_size"]


@pytest.mark.parametrize(
    "patch_size", [[12], [0, 12, 12], [12, 12, 13], [16, 10, 16], [12, 12, 12, 12]]
)
def test_wrong_patch_size(minimum_algorithm_n2v: dict, patch_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_algorithm_n2v["axes"] = "ZYX" if len(patch_size) == 3 else "YX"
    minimum_algorithm_n2v["patch_size"] = patch_size

    with pytest.raises(ValueError):
        DataConfig(**minimum_algorithm_n2v)


def test_set_3d(minimum_algorithm_n2v: dict):
    """Test that 3D can be set."""
    data = DataConfig(**minimum_algorithm_n2v)
    assert "Z" not in data.axes
    assert len(data.patch_size) == 2

    # error if changing Z manually
    with pytest.raises(ValueError):
        data.axes = "ZYX"

    # or patch size
    data = DataConfig(**minimum_algorithm_n2v)
    with pytest.raises(ValueError):
        data.patch_size = [64, 64, 64]

    # set 3D
    data = DataConfig(**minimum_algorithm_n2v)
    data.set_3D("ZYX", [64, 64, 64])
    assert "Z" in data.axes
    assert len(data.patch_size) == 3


@pytest.mark.parametrize(
    "transforms",
    [
        [
            {"name": SupportedTransform.XY_FLIP.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
        [
            {"name": SupportedTransform.XY_FLIP.value},
        ],
        [
            {"name": SupportedTransform.XY_FLIP.value},
            {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
    ],
)
def test_passing_supported_transforms(minimum_algorithm_n2v: dict, transforms):
    """Test that list of supported transforms can be passed."""
    minimum_algorithm_n2v["transforms"] = transforms
    model = DataConfig(**minimum_algorithm_n2v)

    supported = {
        "XYFlip": XYFlipModel,
        "XYRandomRotate90": XYRandomRotate90Model,
        "N2VManipulate": N2VManipulateModel,
    }

    for ind, t in enumerate(transforms):
        assert t["name"] == model.transforms[ind].name
        assert isinstance(model.transforms[ind], supported[t["name"]])


def test_export_to_yaml_float32_stats(tmp_path, minimum_algorithm_n2v: dict):
    """Test exporting and loading the pydantic model when the statistics are
    np.float32."""
    data = DataConfig(**minimum_algorithm_n2v)

    # set np.float32 stats values
    data.set_means_and_stds([np.float32(1234.5678)], [np.float32(21.73)])

    # export to yaml
    config_path = tmp_path / "data_config.yaml"
    with open(config_path, "w") as f:
        # dump configuration
        yaml.dump(data.model_dump(), f, default_flow_style=False, sort_keys=False)

    # load model
    dictionary = yaml.load(config_path.open("r"), Loader=yaml.SafeLoader)
    read_data = DataConfig(**dictionary)
    assert read_data.model_dump() == data.model_dump()

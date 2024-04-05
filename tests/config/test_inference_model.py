import pytest
from albumentations import Compose

from careamics.config.inference_model import InferenceModel
from careamics.config.support import (
    SupportedTransform,
)
from careamics.config.transformations.transform_model import TransformModel
from careamics.transforms import get_all_transforms


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_inference: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_inference["data_type"] = ext

    # instantiate InferenceModel model
    with pytest.raises(ValueError):
        InferenceModel(**minimum_inference)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(minimum_inference: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    minimum_inference["mean"] = mean
    minimum_inference["std"] = std

    prediction_model = InferenceModel(**minimum_inference)
    assert prediction_model.mean == mean
    assert prediction_model.std == std


def test_mean_std_both_specified_or_none(minimum_inference: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    InferenceModel(**minimum_inference)

    # Error if only mean is defined
    minimum_inference["mean"] = 10.4
    with pytest.raises(ValueError):
        InferenceModel(**minimum_inference)

    # Error if only std is defined
    minimum_inference.pop("mean")
    minimum_inference["std"] = 10.4
    with pytest.raises(ValueError):
        InferenceModel(**minimum_inference)

    # No error if both are specified
    minimum_inference["mean"] = 10.4
    minimum_inference["std"] = 10.4
    InferenceModel(**minimum_inference)


def test_set_mean_and_std(minimum_inference: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they None
    mean = 4.07
    std = 14.07
    pred = InferenceModel(**minimum_inference)
    pred.set_mean_and_std(mean, std)
    assert pred.mean == mean
    assert pred.std == std

    # and if they are already set
    minimum_inference["mean"] = 10.4
    minimum_inference["std"] = 3.2
    pred = InferenceModel(**minimum_inference)
    pred.set_mean_and_std(mean, std)
    assert pred.mean == mean
    assert pred.std == std


def test_tile_size(minimum_inference: dict):
    """Test that non-zero even patch size are accepted."""
    # 2D
    prediction_model = InferenceModel(**minimum_inference)

    # 3D
    minimum_inference["tile_size"] = [12, 12, 12]
    minimum_inference["tile_overlap"] = [2, 2, 2]
    minimum_inference["axes"] = "ZYX"

    prediction_model = InferenceModel(**minimum_inference)
    assert prediction_model.tile_size == [12, 12, 12]
    assert prediction_model.tile_overlap == [2, 2, 2]


@pytest.mark.parametrize(
    "tile_size", [[12], [0, 12, 12], [12, 12, 13], [12, 12, 12, 12]]
)
def test_wrong_tile_size(minimum_inference: dict, tile_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_inference["axes"] = "ZYX" if len(tile_size) == 3 else "YX"
    minimum_inference["tile_size"] = tile_size

    with pytest.raises(ValueError):
        InferenceModel(**minimum_inference)


def test_set_3d(minimum_inference: dict):
    """Test that 3D can be set."""
    pred = InferenceModel(**minimum_inference)
    assert "Z" not in pred.axes
    assert len(pred.tile_size) == 2
    assert len(pred.tile_overlap) == 2

    # error if changing Z manually
    with pytest.raises(ValueError):
        pred.axes = "ZYX"

    # or patch size
    pred = InferenceModel(**minimum_inference)
    with pytest.raises(ValueError):
        pred.tile_size = [64, 64, 64]
    
    with pytest.raises(ValueError):
        pred.tile_overlap = [64, 64, 64]

    # set 3D
    pred = InferenceModel(**minimum_inference)
    pred.set_3D("ZYX", [64, 64, 64], [32, 32, 32])
    assert "Z" in pred.axes
    assert len(pred.tile_size) == 3
    assert len(pred.tile_overlap) == 3


@pytest.mark.parametrize("transforms",
    [

        [
            {"name": SupportedTransform.NORMALIZE.value},
        ],
        [
            {"name": SupportedTransform.NORMALIZE.value},
            {"name": SupportedTransform.NDFLIP.value},
            {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        ],
    ]
)
def test_passing_supported_transforms(minimum_inference: dict, transforms):
    """Test that list of supported transforms can be passed."""
    minimum_inference["transforms"] = transforms
    InferenceModel(**minimum_inference)


def test_cannot_pass_n2v_manipulate(minimum_inference: dict):
    """Test that passing N2V pixel manipulate transform raises an error."""
    minimum_inference["transforms"] = [
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    with pytest.raises(ValueError):
        InferenceModel(**minimum_inference)

def test_passing_empty_transforms(minimum_inference: dict):
    """Test that empty list of transforms can be passed."""
    minimum_inference["transforms"] = []
    InferenceModel(**minimum_inference)


def test_passing_incorrect_element(minimum_inference: dict):
    """Test that incorrect element in the list of transforms raises an error (
    e.g. passing un object rather than a string)."""
    minimum_inference["transforms"] = [
        {"name": get_all_transforms()[SupportedTransform.NDFLIP.value]()},
    ]
    with pytest.raises(ValueError):
        InferenceModel(**minimum_inference)


def test_passing_compose_transform(minimum_inference: dict):
    """Test that Compose transform can be passed."""
    minimum_inference["transforms"] = Compose(
        [
            get_all_transforms()[SupportedTransform.NORMALIZE](),
            get_all_transforms()[SupportedTransform.NDFLIP](),
        ]
    )
    InferenceModel(**minimum_inference)


def test_passing_albumentations_transform(minimum_inference: dict):
    """Test passing an albumentation transform with parameters."""
    minimum_inference["transforms"] = [
        {
            "name": "PixelDropout",
            "parameters": {
                "dropout_prob": 0.05,
                "per_channel": True,
            },
        },
    ]
    model = InferenceModel(**minimum_inference)
    assert isinstance(model.transforms[0], TransformModel)

    params = model.transforms[0].parameters.model_dump()
    assert params["dropout_prob"] == 0.05
    assert params["per_channel"] is True

    # check that we can instantiate the transform
    get_all_transforms()[model.transforms[0].name](**params)


def test_mean_and_std_in_normalize(minimum_inference: dict):
    """Test that mean and std are added to the Normalize transform."""
    minimum_inference["mean"] = 10.4
    minimum_inference["std"] = 3.2
    minimum_inference["transforms"] = [
        {"name": SupportedTransform.NORMALIZE.value},
    ]

    data = InferenceModel(**minimum_inference)
    assert data.transforms[0].parameters.mean == 10.4
    assert data.transforms[0].parameters.std == 3.2

import pytest
from albumentations import Compose

from careamics.config.prediction_model import PredictionModel
from careamics.config.transformations.xy_random_rotate90_model import (
    XYRandomRotate90Model
)
from careamics.config.transformations.transform_model import TransformModel
from careamics.config.support import (
    SupportedTransform, SupportedStructAxis, SupportedPixelManipulation
)
from careamics.transforms import get_all_transforms


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_prediction: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_prediction["data_type"] = ext

    # instantiate PredictionModel model
    with pytest.raises(ValueError):
        PredictionModel(**minimum_prediction)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(minimum_prediction: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    minimum_prediction["mean"] = mean
    minimum_prediction["std"] = std

    prediction_model = PredictionModel(**minimum_prediction)
    assert prediction_model.mean == mean
    assert prediction_model.std == std


def test_mean_std_both_specified_or_none(minimum_prediction: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    PredictionModel(**minimum_prediction)

    # Error if only mean is defined
    minimum_prediction["mean"] = 10.4
    with pytest.raises(ValueError):
        PredictionModel(**minimum_prediction)

    # Error if only std is defined
    minimum_prediction.pop("mean")
    minimum_prediction["std"] = 10.4
    with pytest.raises(ValueError):
        PredictionModel(**minimum_prediction)

    # No error if both are specified
    minimum_prediction["mean"] = 10.4
    minimum_prediction["std"] = 10.4
    PredictionModel(**minimum_prediction)


def test_set_mean_and_std(minimum_prediction: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they None
    mean = 4.07
    std = 14.07
    pred = PredictionModel(**minimum_prediction)
    pred.set_mean_and_std(mean, std)
    assert pred.mean == mean
    assert pred.std == std

    # and if they are already set
    minimum_prediction["mean"] = 10.4
    minimum_prediction["std"] = 3.2
    pred = PredictionModel(**minimum_prediction)
    pred.set_mean_and_std(mean, std)
    assert pred.mean == mean
    assert pred.std == std


def test_tile_size(minimum_prediction: dict):
    """Test that non-zero even patch size are accepted."""
    # 2D
    prediction_model = PredictionModel(**minimum_prediction)

    # 3D
    minimum_prediction["tile_size"] = [12, 12, 12]
    minimum_prediction["axes"] = "ZYX"

    prediction_model = PredictionModel(**minimum_prediction)
    assert prediction_model.tile_size == [12, 12, 12]


@pytest.mark.parametrize(
    "tile_size", [[12], [0, 12, 12], [12, 12, 13], [12, 12, 12, 12]]
)
def test_wrong_tile_size(minimum_prediction: dict, tile_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_prediction["axes"] = "ZYX" if len(tile_size) == 3 else "YX"
    minimum_prediction["tile_size"] = tile_size

    with pytest.raises(ValueError):
        PredictionModel(**minimum_prediction)


def test_set_3d(minimum_prediction: dict):
    """Test that 3D can be set."""
    pred = PredictionModel(**minimum_prediction)
    assert "Z" not in pred.axes
    assert len(pred.tile_size) == 2

    # error if changing Z manually
    with pytest.raises(ValueError):
        pred.axes = "ZYX"

    # or patch size
    pred = PredictionModel(**minimum_prediction)
    with pytest.raises(ValueError):
        pred.tile_size = [64, 64, 64]

    # set 3D
    pred = PredictionModel(**minimum_prediction)
    pred.set_3D("ZYX", [64, 64, 64])
    assert "Z" in pred.axes
    assert len(pred.tile_size) == 3


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
def test_passing_supported_transforms(minimum_prediction: dict, transforms):
    """Test that list of supported transforms can be passed."""
    minimum_prediction["transforms"] = transforms
    PredictionModel(**minimum_prediction)


def test_cannot_pass_n2v_manipulate(minimum_prediction: dict):
    """Test that passing N2V pixel manipulate transform raises an error."""
    minimum_prediction["transforms"] = [
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    with pytest.raises(ValueError):
        PredictionModel(**minimum_prediction)

def test_passing_empty_transforms(minimum_prediction: dict):
    """Test that empty list of transforms can be passed."""
    minimum_prediction["transforms"] = []
    PredictionModel(**minimum_prediction)


def test_passing_incorrect_element(minimum_prediction: dict):
    """Test that incorrect element in the list of transforms raises an error (
    e.g. passing un object rather than a string)."""
    minimum_prediction["transforms"] = [
        {"name": get_all_transforms()[SupportedTransform.NDFLIP.value]()},
    ]
    with pytest.raises(ValueError):
        PredictionModel(**minimum_prediction)


def test_passing_compose_transform(minimum_prediction: dict):
    """Test that Compose transform can be passed."""
    minimum_prediction["transforms"] = Compose(
        [
            get_all_transforms()[SupportedTransform.NORMALIZE](),
            get_all_transforms()[SupportedTransform.NDFLIP](),
        ]
    )
    PredictionModel(**minimum_prediction)


def test_passing_albumentations_transform(minimum_prediction: dict):
    """Test passing an albumentation transform with parameters."""
    minimum_prediction["transforms"] = [
        {
            "name": "PixelDropout",
            "parameters": {
                "dropout_prob": 0.05, 
                "per_channel": True,
            },
        },
    ]
    model = PredictionModel(**minimum_prediction)
    assert isinstance(model.transforms[0], TransformModel)
    
    params = model.transforms[0].parameters.model_dump()
    assert params["dropout_prob"] == 0.05
    assert params["per_channel"] is True

    # check that we can instantiate the transform
    get_all_transforms()[model.transforms[0].name](**params)

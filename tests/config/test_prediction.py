import pytest
from pydantic import conlist

from careamics_restoration.config.prediction import Prediction


@pytest.mark.parametrize("tile_shape", [[4, 4], [6, 4, 4], [32, 96]])
def test_prediction_tile_shape(complete_config: dict, tile_shape: conlist):
    """Test tile shape greater than 1 and divisible by 2."""
    prediction = complete_config["prediction"]
    prediction["tile_shape"] = tile_shape
    prediction["overlaps"] = [2 for _ in tile_shape]

    prediction = Prediction(**prediction)
    assert prediction.tile_shape == tile_shape


@pytest.mark.parametrize(
    "tile_shape",
    [
        [4],
        [4, 4, 4, 4],
        [33, 32],
        None,
    ],
)
def test_prediction_wrong_tile_shape(complete_config: dict, tile_shape: conlist):
    """Test that wrong tile shape cause an error."""
    prediction = complete_config["prediction"]
    prediction["tile_shape"] = tile_shape
    prediction["overlaps"] = [2, 2]

    with pytest.raises(ValueError):
        Prediction(**prediction)


@pytest.mark.parametrize(
    "tile_shape, overlaps",
    [
        ([96, 96], [16, 2]),
        ([96, 96, 96], [2, 32, 94]),
    ],
)
def test_prediction_overlaps(
    complete_config: dict, tile_shape: conlist, overlaps: conlist
):
    """Test overlaps greater than 1 and divisible by 2, and smaller than tiles."""
    prediction = complete_config["prediction"]
    prediction["tile_shape"] = tile_shape
    prediction["overlaps"] = overlaps

    prediction = Prediction(**prediction)
    assert prediction.tile_shape == tile_shape


@pytest.mark.parametrize(
    "tile_shape, overlaps",
    [
        ([96, 96], [2]),
        ([96, 96], [1, 1]),
        ([96, 96], [0, 0]),
        ([96, 96, 96], [32, 94]),
        ([96, 96, 96], [96, 94, 94]),
        ([96, 96], None),
    ],
)
def test_prediction_wrong_overlaps(
    complete_config: dict, tile_shape: conlist, overlaps: conlist
):
    """Test that wrong tile shape cause an error."""
    prediction = complete_config["prediction"]
    prediction["tile_shape"] = tile_shape
    prediction["overlaps"] = overlaps

    with pytest.raises(ValueError):
        Prediction(**prediction)


def test_prediction_no_tiling():
    """Test that no tiling can be specified without passing other paramters."""
    prediction = Prediction(use_tiling=False)
    assert prediction.tile_shape is None
    assert prediction.overlaps is None


def test_prediction_tiling_without_parameters():
    """Check that tiling True requires both optional parameters."""
    with pytest.raises(ValueError):
        Prediction(use_tiling=True)

    with pytest.raises(ValueError):
        Prediction(use_tiling=True, tile_shape=[96, 96])

    with pytest.raises(ValueError):
        Prediction(use_tiling=True, overlaps=[96, 96])


def test_prediction_to_dict(complete_config):
    """Test that to_dict method works."""
    prediction_dict = Prediction(**complete_config["prediction"]).model_dump()
    assert prediction_dict == complete_config["prediction"]


def test_prediction_to_dict_no_optionals():
    """Test that model_dump method works without the optional parameters."""
    prediction_dict = Prediction(use_tiling=False).model_dump()
    assert prediction_dict == {"use_tiling": False}

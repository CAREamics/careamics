import pytest

from careamics.config.transform_model import TransformModel, ALL_TRANSFORMS


def test_manipulateN2V_in_transforms():
    """Test that ManipulateN2V is in ALL_TRANSFORMS."""
    assert "ManipulateN2V" in ALL_TRANSFORMS


def test_normalize_without_targets_in_transforms():
    """Test that NormalizeWithoutTarget is in ALL_TRANSFORMS."""
    assert "NormalizeWithoutTarget" in ALL_TRANSFORMS


@pytest.mark.parametrize("name, parameters", 
    [
        ("flip", {}),
        ("flip", {"p": 0.5}),
        ("ManipulateN2V", {"masked_pixel_percentage": 0.2, "roi_size": 11}),
        ("ManipulateN2V", {}),
    ]
)
def test_transform(name, parameters):
    TransformModel(name=name, parameters=parameters)


@pytest.mark.parametrize("name, parameters", 
    [
        ("flippy", {"p": 0.5}),
        ("flip", {"ps": 0.5}),
    ]
)
def test_transform_wrong_values(name, parameters):
    with pytest.raises(ValueError):
        TransformModel(name=name, parameters=parameters)
        

@pytest.mark.parametrize("roi_size", [5, 9, 15])
def test_parameters_roi_size(roi_size: int):
    """Test that Algorithm accepts roi_size as an even number within the
    range [3, 21]."""
    # complete_config["algorithm"]["masking_strategy"]["parameters"][
    #     "roi_size"
    # ] = roi_size
    # algorithm = Algorithm(**complete_config["algorithm"])
    # assert algorithm.masking_strategy.parameters["roi_size"] == roi_size
    # TODO
    pass

@pytest.mark.parametrize("roi_size", [2, 4, 23])
def test_parameters_wrong_roi_size(roi_size: int):
    """Test that wrong num_channels_init cause an error."""
    # complete_config["algorithm"]["masking_strategy"]["parameters"][
    #     "roi_size"
    # ] = roi_size
    # with pytest.raises(ValueError):
    #     Algorithm(**complete_config["algorithm"])
    # TODO
    pass


@pytest.mark.parametrize("masked_pixel_percentage", [0.1, 0.2, 5, 20])
def test_masked_pixel_percentage(masked_pixel_percentage: float):
    """Test that Algorithm accepts the minimum configuration."""
    # algorithm = complete_config["algorithm"]
    # algorithm["masking_strategy"]["parameters"][
    #     "masked_pixel_percentage"
    # ] = masked_pixel_percentage

    # algo = Algorithm(**algorithm)
    # assert (
    #     algo.masking_strategy.parameters["masked_pixel_percentage"]
    #     == masked_pixel_percentage
    # )
    # TODO
    pass


@pytest.mark.parametrize("masked_pixel_percentage", [0.01, 21])
def test_wrong_masked_pixel_percentage(
    masked_pixel_percentage: float
):
    """Test that Algorithm accepts the minimum configuration."""
    # algorithm = complete_config["algorithm"]["masking_strategy"]["parameters"]
    # algorithm["masked_pixel_percentage"] = masked_pixel_percentage

    # with pytest.raises(ValueError):
    #     Algorithm(**algorithm)
    # TODO
    pass
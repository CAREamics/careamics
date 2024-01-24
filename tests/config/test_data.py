import pytest

from careamics.config.data import Data, SupportedExtension

    
@pytest.mark.parametrize("ext", ["tiff", "tif", "TIFF", "TIF", ".TIF"])
def test_supported_extensions_case_insensitive(ext: str):
    """Test that SupportedExtension enum accepts all extensions in upper
    cases and with ."""
    sup_ext = SupportedExtension(ext)

    new_ext = ext.lower()
    if ext.startswith("."):
        new_ext = new_ext[1:]

    assert sup_ext.value == new_ext


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_data: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_data["data_format"] = ext

    # instantiate Data model
    with pytest.raises(ValueError):
        Data(**minimum_data)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(minimum_data: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    minimum_data["mean"] = mean
    minimum_data["std"] = std

    data_model = Data(**minimum_data)
    assert data_model.mean == mean
    assert data_model.std == std


def test_mean_std_negative(minimum_data: dict):
    """Test that negative mean and std are not accepted."""
    minimum_data["mean"] = -1
    minimum_data["std"] = 10.4

    with pytest.raises(ValueError):
        Data(**minimum_data)

    minimum_data["mean"] = 10.4
    minimum_data["std"] = -1

    with pytest.raises(ValueError):
        Data(**minimum_data)


def test_mean_std_both_specified_or_none(minimum_data: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    Data(**minimum_data)

    # No error if mean is defined
    minimum_data["mean"] = 10.4
    Data(**minimum_data)

    # Error if only std is defined
    minimum_data.pop("mean")
    minimum_data["std"] = 10.4

    with pytest.raises(ValueError):
        Data(**minimum_data)


def test_set_mean_and_std(complete_config: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they are already set
    data = Data(**minimum_data)
    data.set_mean_and_std(4.07, 14.07)

    # and if they are both None
    minimum_data.pop("mean")
    minimum_data.pop("std")
    data = Data(**minimum_data)
    data.set_mean_and_std(10.4, 0.5)


def test_wrong_values_by_assigment(complete_config: dict):
    """Test that wrong values are not accepted through assignment."""
    data_model = Data(**minimum_data)

    # in memory
    data_model.in_memory = minimum_data["in_memory"]
    with pytest.raises(ValueError):
        data_model.in_memory = "Trues"

    # data format
    data_model.data_format = minimum_data["data_format"]  # check assignment
    with pytest.raises(ValueError):
        data_model.data_format = "png"

    # axes
    data_model.axes = minimum_data["axes"]
    with pytest.raises(ValueError):
        data_model.axes = "-YX"

    # mean
    data_model.mean = minimum_data["mean"]
    with pytest.raises(ValueError):
        data_model.mean = -1

    # std
    data_model.std = minimum_data["std"]
    with pytest.raises(ValueError):
        data_model.std = -1


def test_data_to_dict_minimum(minimum_config: dict):
    """Test that export to dict does not include None values and Paths.

    In the minimum config, only training+validation should be defined, all the other
    paths are None."""
    data_minimum = Data(**minimum_config["data"]).model_dump()
    assert data_minimum == minimum_config["data"]

    assert "in_memory" in data_minimum.keys()
    assert "data_format" in data_minimum.keys()
    assert "axes" in data_minimum.keys()
    assert "mean" not in data_minimum.keys()
    assert "std" not in data_minimum.keys()


def test_data_to_dict_complete(complete_config: dict):
    """Test that export to dict does not include None values and Paths."""
    data_complete = Data(**minimum_data).model_dump()
    assert data_complete == minimum_data

    assert "in_memory" in data_complete.keys()
    assert "data_format" in data_complete.keys()
    assert "axes" in data_complete.keys()
    assert "mean" in data_complete.keys()
    assert "std" in data_complete.keys()

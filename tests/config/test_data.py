import pytest

from careamics.config.data import Data


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


def test_set_mean_and_std(minimum_data: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they are already set
    data = Data(**minimum_data)
    data.set_mean_and_std(4.07, 14.07)

    # and if they are both None
    minimum_data["mean"] = 10.4
    minimum_data["std"] = 3.2
    data = Data(**minimum_data)
    data.set_mean_and_std(10.4, 0.5)


def test_patch_size(minimum_data: dict):
    """Test that non-zero even patch size are accepted."""
    minimum_data["patch_size"] = [12, 12, 12]

    data_model = Data(**minimum_data)
    assert data_model.patch_size == [12, 12, 12]


@pytest.mark.parametrize("patch_size",
    [
        [12],
        [0, 12, 12],
        [12, 12, 13],
        [12, 12, 12, 12]
    ]
)
def test_wrong_patch_size(minimum_data: dict, patch_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_data["patch_size"] = patch_size

    with pytest.raises(ValueError):
        Data(**minimum_data)


# TODO transforms validation tests

def test_wrong_values_by_assigment(minimum_data: dict):
    """Test that wrong values are not accepted through assignment."""
    data_model = Data(**minimum_data)

    # in memory
    data_model.in_memory = False
    with pytest.raises(ValueError):
        data_model.in_memory = "Trues"

    # data format
    data_model.extension = "tiff"
    with pytest.raises(ValueError):
        data_model.extension = "png"

    # axes
    data_model.axes = "SZYX"
    with pytest.raises(ValueError):
        data_model.axes = "-YX"

    # mean
    data_model.mean = 12
    with pytest.raises(ValueError):
        data_model.mean = -1

    # std
    data_model.std = 3.6
    with pytest.raises(ValueError):
        data_model.std = -1

    # patch size
    data_model.patch_size = [12, 12, 12]
    with pytest.raises(ValueError):
        data_model.patch_size = [12]

    # TODO transforms


def test_data_to_dict_minimum(minimum_data: dict):
    """Test that export to dict does not include optional values."""
    data_minimum = Data(**minimum_data).model_dump()
    assert data_minimum == minimum_data

import pytest
from albumentations import Compose

from careamics.config.data_model import DataModel
from careamics.config.transformations.xy_random_rotate90_model import (
    XYRandomRotate90Model
)
from careamics.config.transformations.transform_model import TransformModel
from careamics.config.support import (
    SupportedTransform, SupportedStructAxis, SupportedPixelManipulation
)
from careamics.transforms import get_all_transforms


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_data: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_data["data_type"] = ext

    # instantiate DataModel model
    with pytest.raises(ValueError):
        DataModel(**minimum_data)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(minimum_data: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    minimum_data["mean"] = mean
    minimum_data["std"] = std

    data_model = DataModel(**minimum_data)
    assert data_model.mean == mean
    assert data_model.std == std


def test_mean_std_both_specified_or_none(minimum_data: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    DataModel(**minimum_data)

    # Error if only mean is defined
    minimum_data["mean"] = 10.4
    with pytest.raises(ValueError):
        DataModel(**minimum_data)

    # Error if only std is defined
    minimum_data.pop("mean")
    minimum_data["std"] = 10.4
    with pytest.raises(ValueError):
        DataModel(**minimum_data)

    # No error if both are specified
    minimum_data["mean"] = 10.4
    minimum_data["std"] = 10.4
    DataModel(**minimum_data)


def test_set_mean_and_std(minimum_data: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they None
    mean = 4.07
    std = 14.07
    data = DataModel(**minimum_data)
    data.set_mean_and_std(mean, std)
    assert data.mean == mean
    assert data.std == std

    # and if they are already set
    minimum_data["mean"] = 10.4
    minimum_data["std"] = 3.2
    data = DataModel(**minimum_data)
    data.set_mean_and_std(mean, std)
    assert data.mean == mean
    assert data.std == std


def test_patch_size(minimum_data: dict):
    """Test that non-zero even patch size are accepted."""
    # 2D
    data_model = DataModel(**minimum_data)

    # 3D
    minimum_data["patch_size"] = [12, 12, 12]
    minimum_data["axes"] = "ZYX"

    data_model = DataModel(**minimum_data)
    assert data_model.patch_size == [12, 12, 12]


@pytest.mark.parametrize(
    "patch_size", [[12], [0, 12, 12], [12, 12, 13], [12, 12, 12, 12]]
)
def test_wrong_patch_size(minimum_data: dict, patch_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_data["axes"] = "ZYX" if len(patch_size) == 3 else "YX"
    minimum_data["patch_size"] = patch_size

    with pytest.raises(ValueError):
        DataModel(**minimum_data)


def test_set_3d(minimum_data: dict):
    """Test that 3D can be set."""
    data = DataModel(**minimum_data)
    assert "Z" not in data.axes
    assert len(data.patch_size) == 2

    # error if changing Z manually
    with pytest.raises(ValueError):
        data.axes = "ZYX"

    # or patch size
    data = DataModel(**minimum_data)
    with pytest.raises(ValueError):
        data.patch_size = [64, 64, 64]

    # set 3D
    data = DataModel(**minimum_data)
    data.set_3D("ZYX", [64, 64, 64])
    assert "Z" in data.axes
    assert len(data.patch_size) == 3


@pytest.mark.parametrize("transforms",
    [
        [
            {"name": SupportedTransform.NDFLIP.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
        [
            {"name": SupportedTransform.NDFLIP.value},
        ],
        [
            {"name": SupportedTransform.NORMALIZE.value},
            {"name": SupportedTransform.NDFLIP.value},
            {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
    ]
)
def test_passing_supported_transforms(minimum_data: dict, transforms):
    """Test that list of supported transforms can be passed."""
    minimum_data["transforms"] = transforms
    DataModel(**minimum_data)


@pytest.mark.parametrize("transforms",
    [
        [
            {"name": SupportedTransform.N2V_MANIPULATE.value},
            {"name": SupportedTransform.NDFLIP.value},
        ],
        [
            {"name": SupportedTransform.N2V_MANIPULATE.value},
        ],
        [
            {"name": SupportedTransform.NORMALIZE.value},
            {"name": SupportedTransform.NDFLIP.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
            {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        ],
    ]
)
def test_n2vmanipulate_last_transform(minimum_data: dict, transforms):
    """Test that N2V Manipulate is moved to the last position if it is not."""
    minimum_data["transforms"] = transforms
    model = DataModel(**minimum_data)
    assert model.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value


def test_multiple_n2v_manipulate(minimum_data: dict):
    """Test that passing multiple n2v manipulate raises an error."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.N2V_MANIPULATE.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    with pytest.raises(ValueError):
        DataModel(**minimum_data)


def test_remove_n2v_manipulate(minimum_data: dict):
    """Test that N2V Manipulate can be removed."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.NDFLIP.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    model = DataModel(**minimum_data)
    model.remove_n2v_manipulate()
    assert len(model.transforms) == 1
    assert model.transforms[-1].name == SupportedTransform.NDFLIP.value


def test_add_n2v_manipulate(minimum_data: dict):
    """Test that N2V Manipulate can be added."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.NDFLIP.value},
    ]
    model = DataModel(**minimum_data)
    model.add_n2v_manipulate()
    assert len(model.transforms) == 2
    assert model.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value

    # test that adding twice doesn't change anything
    model.add_n2v_manipulate()
    assert len(model.transforms) == 2
    assert model.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value


def test_correct_transform_parameters(minimum_data: dict):
    """Test that the transforms have the correct parameters.
    
    This is important to know that the transforms are not all instantiated as
    a generic transform.
    """
    minimum_data["transforms"] = [
        {"name": SupportedTransform.NORMALIZE.value},
        {"name": SupportedTransform.NDFLIP.value},
        {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    model = DataModel(**minimum_data)

    # Normalize
    params = model.transforms[0].parameters.model_dump()
    assert "mean" in params
    assert "std" in params
    assert "max_pixel_value" in params

    # NDFlip
    params = model.transforms[1].parameters.model_dump()
    assert "p" in params
    assert "is_3D" in params
    assert "flip_z" in params

    # XYRandomRotate90
    params = model.transforms[2].parameters.model_dump()
    assert "p" in params
    assert "is_3D" in params
    assert isinstance(model.transforms[2], XYRandomRotate90Model)

    # N2VManipulate
    params = model.transforms[3].parameters.model_dump()
    assert "roi_size" in params
    assert "masked_pixel_percentage" in params
    assert "strategy" in params
    assert "struct_mask_axis" in params
    assert "struct_mask_span" in params


def test_passing_empty_transforms(minimum_data: dict):
    """Test that empty list of transforms can be passed."""
    minimum_data["transforms"] = []
    DataModel(**minimum_data)


def test_passing_incorrect_element(minimum_data: dict):
    """Test that incorrect element in the list of transforms raises an error (
    e.g. passing un object rather than a string)."""
    minimum_data["transforms"] = [
        {"name": get_all_transforms()[SupportedTransform.NDFLIP.value]()},
    ]
    with pytest.raises(ValueError):
        DataModel(**minimum_data)


def test_passing_compose_transform(minimum_data: dict):
    """Test that Compose transform can be passed."""
    minimum_data["transforms"] = Compose(
        [
            get_all_transforms()[SupportedTransform.NDFLIP](),
            get_all_transforms()[SupportedTransform.N2V_MANIPULATE](),
        ]
    )
    DataModel(**minimum_data)


def test_passing_albumentations_transform(minimum_data: dict):
    """Test passing an albumentation transform with parameters."""
    minimum_data["transforms"] = [
        {
            "name": "PixelDropout",
            "parameters": {
                "dropout_prob": 0.05, 
                "per_channel": True,
            },
        },
    ]
    model = DataModel(**minimum_data)
    assert isinstance(model.transforms[0], TransformModel)
    
    params = model.transforms[0].parameters.model_dump()
    assert params["dropout_prob"] == 0.05
    assert params["per_channel"] is True

    # check that we can instantiate the transform
    get_all_transforms()[model.transforms[0].name](**params)


def test_3D_and_transforms(minimum_data: dict):
    """Test that NDFlip is corrected if the data is 3D."""
    minimum_data["transforms"] = [
        {
            "name": SupportedTransform.NDFLIP.value,
            "parameters": {
                "is_3D": True,
                "flip_z": True,
            },
        },
        {
            "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            "parameters": {
                "is_3D": True,
            },
        },
    ]
    data = DataModel(**minimum_data)
    assert data.transforms[0].parameters.is_3D is False
    assert data.transforms[1].parameters.is_3D is False

    # change to 3D
    data.set_3D("ZYX", [64, 64, 64])
    data.transforms[0].parameters.is_3D = True
    data.transforms[1].parameters.is_3D = True


def test_set_n2v_strategy(minimum_data: dict):
    """Test that the N2V strategy can be set."""
    uniform = SupportedPixelManipulation.UNIFORM.value
    median = SupportedPixelManipulation.MEDIAN.value

    data = DataModel(**minimum_data)
    assert data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert data.transforms[-1].parameters.strategy == uniform

    data.set_N2V2_strategy(median)
    assert data.transforms[-1].parameters.strategy == median

    data.set_N2V2_strategy(uniform)
    assert data.transforms[-1].parameters.strategy == uniform


def test_set_n2v_strategy_wrong_value(minimum_data: dict):
    """Test that passing a wrong strategy raises an error."""
    data = DataModel(**minimum_data)
    with pytest.raises(ValueError):
        data.set_N2V2_strategy("wrong_value")


def test_set_struct_mask(minimum_data: dict):
    """Test that the struct mask can be set."""
    none = SupportedStructAxis.NONE.value
    vertical = SupportedStructAxis.VERTICAL.value
    horizontal = SupportedStructAxis.HORIZONTAL.value

    data = DataModel(**minimum_data)
    assert data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert data.transforms[-1].parameters.struct_mask_axis == none
    assert data.transforms[-1].parameters.struct_mask_span == 5

    data.set_structN2V_mask(vertical, 3)
    assert data.transforms[-1].parameters.struct_mask_axis == vertical
    assert data.transforms[-1].parameters.struct_mask_span == 3

    data.set_structN2V_mask(horizontal, 7)
    assert data.transforms[-1].parameters.struct_mask_axis == horizontal
    assert data.transforms[-1].parameters.struct_mask_span == 7

    data.set_structN2V_mask(none, 11)
    assert data.transforms[-1].parameters.struct_mask_axis == none
    assert data.transforms[-1].parameters.struct_mask_span == 11


def test_set_struct_mask_wrong_value(minimum_data: dict):
    """Test that passing a wrong struct mask axis raises an error."""
    data = DataModel(**minimum_data)
    with pytest.raises(ValueError):
        data.set_structN2V_mask("wrong_value", 3)

    with pytest.raises(ValueError):
        data.set_structN2V_mask(SupportedStructAxis.VERTICAL.value, 1)
    
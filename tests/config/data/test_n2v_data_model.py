import pytest

from careamics.config.data import N2VDataConfig
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
)


def test_error_no_manipulate(minimum_data: dict):
    """Test that an error is raised if no N2VManipulate transform is passed."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.XY_FLIP.value},
        {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
    ]
    with pytest.raises(ValueError):
        N2VDataConfig(**minimum_data)


@pytest.mark.parametrize(
    "transforms",
    [
        [
            {"name": SupportedTransform.N2V_MANIPULATE.value},
            {"name": SupportedTransform.XY_FLIP.value},
        ],
        [
            {"name": SupportedTransform.XY_FLIP.value},
            {"name": SupportedTransform.N2V_MANIPULATE.value},
            {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        ],
    ],
)
def test_n2vmanipulate_not_last_transform(minimum_data: dict, transforms):
    """Test that N2V Manipulate not in the last position raises an error."""
    minimum_data["transforms"] = transforms
    with pytest.raises(ValueError):
        N2VDataConfig(**minimum_data)


def test_multiple_n2v_manipulate(minimum_data: dict):
    """Test that passing multiple n2v manipulate raises an error."""
    minimum_data["transforms"] = [
        {"name": SupportedTransform.N2V_MANIPULATE.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    with pytest.raises(ValueError):
        N2VDataConfig(**minimum_data)


def test_correct_transform_parameters(minimum_data: dict):
    """Test that the transforms have the correct parameters.

    This is important to know that the transforms are not all instantiated as
    a generic transform.
    """
    minimum_data["transforms"] = [
        {"name": SupportedTransform.XY_FLIP.value},
        {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    model = N2VDataConfig(**minimum_data)

    # N2VManipulate
    params = model.transforms[-1].model_dump()
    assert "roi_size" in params
    assert "masked_pixel_percentage" in params
    assert "strategy" in params
    assert "struct_mask_axis" in params
    assert "struct_mask_span" in params


def test_set_n2v_strategy(minimum_data: dict):
    """Test that the N2V strategy can be set."""
    uniform = SupportedPixelManipulation.UNIFORM.value
    median = SupportedPixelManipulation.MEDIAN.value

    data = N2VDataConfig(**minimum_data)
    assert data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert data.transforms[-1].strategy == uniform

    data.set_masking_strategy(median)
    assert data.transforms[-1].strategy == median

    data.set_masking_strategy(uniform)
    assert data.transforms[-1].strategy == uniform


def test_set_n2v_strategy_wrong_value(minimum_data: dict):
    """Test that passing a wrong strategy raises an error."""
    data = N2VDataConfig(**minimum_data)
    with pytest.raises(ValueError):
        data.set_masking_strategy("wrong_value")


def test_set_struct_mask(minimum_data: dict):
    """Test that the struct mask can be set."""
    none = SupportedStructAxis.NONE.value
    vertical = SupportedStructAxis.VERTICAL.value
    horizontal = SupportedStructAxis.HORIZONTAL.value

    data = N2VDataConfig(**minimum_data)
    assert data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert data.transforms[-1].struct_mask_axis == none
    assert data.transforms[-1].struct_mask_span == 5

    data.set_structN2V_mask(vertical, 3)
    assert data.transforms[-1].struct_mask_axis == vertical
    assert data.transforms[-1].struct_mask_span == 3

    data.set_structN2V_mask(horizontal, 7)
    assert data.transforms[-1].struct_mask_axis == horizontal
    assert data.transforms[-1].struct_mask_span == 7

    data.set_structN2V_mask(none, 11)
    assert data.transforms[-1].struct_mask_axis == none
    assert data.transforms[-1].struct_mask_span == 11


def test_set_struct_mask_wrong_value(minimum_data: dict):
    """Test that passing a wrong struct mask axis raises an error."""
    data = N2VDataConfig(**minimum_data)
    with pytest.raises(ValueError):
        data.set_structN2V_mask("wrong_value", 3)

    with pytest.raises(ValueError):
        data.set_structN2V_mask(SupportedStructAxis.VERTICAL.value, 1)

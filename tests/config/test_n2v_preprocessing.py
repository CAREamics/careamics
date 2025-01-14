import pytest

from careamics.config import FCNAlgorithmConfig
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
)
from careamics.transforms import get_all_transforms


def test_correct_transform_parameters(minimum_algorithm_n2v: dict):
    """Test that the transforms have the correct parameters.

    This is important to know that the transforms are not all instantiated as
    a generic transform.
    """
    minimum_algorithm_n2v["transforms"] = [
        {"name": SupportedTransform.XY_FLIP.value},
        {"name": SupportedTransform.XY_RANDOM_ROTATE90.value},
        {"name": SupportedTransform.N2V_MANIPULATE.value},
    ]
    model = FCNAlgorithmConfig(**minimum_algorithm_n2v)

    # N2VManipulate
    params = model.transforms[-1].model_dump()
    assert "roi_size" in params
    assert "masked_pixel_percentage" in params
    assert "strategy" in params
    assert "struct_mask_axis" in params
    assert "struct_mask_span" in params


def test_passing_empty_transforms(minimum_algorithm_n2v: dict):
    """Test that empty list of transforms can be passed."""
    minimum_algorithm_n2v["transforms"] = []
    FCNAlgorithmConfig(**minimum_algorithm_n2v)


def test_passing_incorrect_element(minimum_algorithm_n2v: dict):
    """Test that incorrect element in the list of transforms raises an error (
    e.g. passing un object rather than a string)."""
    minimum_algorithm_n2v["transforms"] = [
        {"name": get_all_transforms()[SupportedTransform.XY_FLIP.value]()},
    ]
    with pytest.raises(ValueError):
        FCNAlgorithmConfig(**minimum_algorithm_n2v)


def test_set_n2v_strategy(minimum_algorithm_n2v: dict):
    """Test that the N2V strategy can be set."""
    uniform = SupportedPixelManipulation.UNIFORM.value
    median = SupportedPixelManipulation.MEDIAN.value

    data = FCNAlgorithmConfig(**minimum_algorithm_n2v)
    assert data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert data.transforms[-1].strategy == uniform

    data.set_N2V2_strategy(median)
    assert data.transforms[-1].strategy == median

    data.set_N2V2_strategy(uniform)
    assert data.transforms[-1].strategy == uniform


def test_set_n2v_strategy_wrong_value(minimum_algorithm_n2v: dict):
    """Test that passing a wrong strategy raises an error."""
    data = FCNAlgorithmConfig(**minimum_algorithm_n2v)
    with pytest.raises(ValueError):
        data.set_N2V2_strategy("wrong_value")


def test_set_struct_mask(minimum_algorithm_n2v: dict):
    """Test that the struct mask can be set."""
    none = SupportedStructAxis.NONE.value
    vertical = SupportedStructAxis.VERTICAL.value
    horizontal = SupportedStructAxis.HORIZONTAL.value

    data = FCNAlgorithmConfig(**minimum_algorithm_n2v)
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


def test_set_struct_mask_wrong_value(minimum_algorithm_n2v: dict):
    """Test that passing a wrong struct mask axis raises an error."""
    data = FCNAlgorithmConfig(**minimum_algorithm_n2v)
    with pytest.raises(ValueError):
        data.set_structN2V_mask("wrong_value", 3)

    with pytest.raises(ValueError):
        data.set_structN2V_mask(SupportedStructAxis.VERTICAL.value, 1)

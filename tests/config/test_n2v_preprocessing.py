import pytest

from careamics.config import N2VAlgorithm
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
    model = N2VAlgorithm(**minimum_algorithm_n2v)

    # N2VManipulate
    params = model.n2v_masking.model_dump()
    assert "roi_size" in params
    assert "masked_pixel_percentage" in params
    assert "strategy" in params
    assert "struct_mask_axis" in params
    assert "struct_mask_span" in params


def test_passing_incorrect_element(minimum_algorithm_n2v: dict):
    """Test that incorrect element in the list of transforms raises an error (
    e.g. passing un object rather than a string)."""
    minimum_algorithm_n2v["n2v_masking"] = {
        "name": get_all_transforms()[SupportedTransform.XY_FLIP.value]()
    }
    with pytest.raises(ValueError):
        N2VAlgorithm(**minimum_algorithm_n2v)


def test_set_n2v_strategy(minimum_algorithm_n2v: dict):
    """Test that the N2V strategy can be set."""
    uniform = SupportedPixelManipulation.UNIFORM.value
    median = SupportedPixelManipulation.MEDIAN.value

    model = N2VAlgorithm(**minimum_algorithm_n2v)
    assert model.n2v_masking.name == SupportedTransform.N2V_MANIPULATE.value
    assert model.n2v_masking.strategy == uniform

    model.n2v_masking.strategy = median
    assert model.n2v_masking.strategy == median

    model.n2v_masking.strategy = uniform
    assert model.n2v_masking.strategy == uniform

    # Check that the strategy is set to median if n2v2 is True
    minimum_algorithm_n2v["model"]["n2v2"] = True
    minimum_algorithm_n2v["n2v_masking"] = {
        "name": SupportedTransform.N2V_MANIPULATE.value,
        "strategy": uniform,
    }
    model = N2VAlgorithm(**minimum_algorithm_n2v)
    assert model.n2v_masking.strategy == median


def test_set_struct_mask(minimum_algorithm_n2v: dict):
    """Test that the struct mask can be set."""
    none = SupportedStructAxis.NONE.value
    vertical = SupportedStructAxis.VERTICAL.value
    horizontal = SupportedStructAxis.HORIZONTAL.value

    model = N2VAlgorithm(**minimum_algorithm_n2v)
    assert model.n2v_masking.name == SupportedTransform.N2V_MANIPULATE.value
    assert model.n2v_masking.struct_mask_axis == none
    assert model.n2v_masking.struct_mask_span == 5

    model.n2v_masking.struct_mask_axis = vertical
    model.n2v_masking. struct_mask_span = 3
    assert model.n2v_masking.struct_mask_axis == vertical
    assert model.n2v_masking.struct_mask_span == 3

    model.n2v_masking.struct_mask_axis = horizontal
    model.n2v_masking.struct_mask_span = 7
    assert model.n2v_masking.struct_mask_axis == horizontal
    assert model.n2v_masking.struct_mask_span == 7

    model.n2v_masking.struct_mask_axis = none
    model.n2v_masking.struct_mask_span = 11
    assert model.n2v_masking.struct_mask_axis == none
    assert model.n2v_masking.struct_mask_span == 11


def test_set_struct_mask_wrong_value(minimum_algorithm_n2v: dict):
    """Test that passing a wrong struct mask axis raises an error."""
    model = N2VAlgorithm(**minimum_algorithm_n2v)
    with pytest.raises(ValueError):
        model.n2v_masking.struct_mask_axis = "wrong_value"

    with pytest.raises(ValueError):
        model.n2v_masking.struct_mask_span = 1

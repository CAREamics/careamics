import pytest

from careamics.config import N2VAlgorithm
from careamics.config.support import (
    SupportedStructAxis,
    SupportedTransform,
)
from careamics.transforms import get_all_transforms

# TODO name is confusing, and the tests are probably not in the right place. They should
# probably be in the N2VManipulate or N2VAlgorithm test file.


def test_correct_transform_parameters(minimum_algorithm_n2v: dict):
    """Test that the transforms have the correct parameters.

    This is important to know that the transforms are not all instantiated as
    a generic transform.
    """
    model = N2VAlgorithm(**minimum_algorithm_n2v)

    # N2VManipulate
    params = model.n2v_config.model_dump()
    assert "roi_size" in params
    assert "masked_pixel_percentage" in params
    assert "strategy" in params
    assert "struct_mask_axis" in params
    assert "struct_mask_span" in params


def test_passing_incorrect_element(minimum_algorithm_n2v: dict):
    """Test that incorrect element in the list of transforms raises an error (
    e.g. passing un object rather than a string)."""
    minimum_algorithm_n2v["n2v_config"] = {
        "name": get_all_transforms()[SupportedTransform.XY_FLIP.value]()
    }
    with pytest.raises(ValueError):
        N2VAlgorithm(**minimum_algorithm_n2v)


def test_set_struct_mask(minimum_algorithm_n2v: dict):
    """Test that the struct mask can be set."""
    none = SupportedStructAxis.NONE.value
    vertical = SupportedStructAxis.VERTICAL.value
    horizontal = SupportedStructAxis.HORIZONTAL.value

    model = N2VAlgorithm(**minimum_algorithm_n2v)
    assert model.n2v_config.name == SupportedTransform.N2V_MANIPULATE.value
    assert model.n2v_config.struct_mask_axis == none
    assert model.n2v_config.struct_mask_span == 5

    model.n2v_config.struct_mask_axis = vertical
    model.n2v_config.struct_mask_span = 3
    assert model.n2v_config.struct_mask_axis == vertical
    assert model.n2v_config.struct_mask_span == 3

    model.n2v_config.struct_mask_axis = horizontal
    model.n2v_config.struct_mask_span = 7
    assert model.n2v_config.struct_mask_axis == horizontal
    assert model.n2v_config.struct_mask_span == 7

    model.n2v_config.struct_mask_axis = none
    model.n2v_config.struct_mask_span = 11
    assert model.n2v_config.struct_mask_axis == none
    assert model.n2v_config.struct_mask_span == 11


def test_set_struct_mask_wrong_value(minimum_algorithm_n2v: dict):
    """Test that passing a wrong struct mask axis raises an error."""
    model = N2VAlgorithm(**minimum_algorithm_n2v)
    with pytest.raises(ValueError):
        model.n2v_config.struct_mask_axis = "wrong_value"

    with pytest.raises(ValueError):
        model.n2v_config.struct_mask_span = 1

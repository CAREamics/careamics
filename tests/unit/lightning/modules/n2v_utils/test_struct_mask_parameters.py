import pytest

from careamics.config.support import SupportedStructAxis
from careamics.lightning.modules.n2v_utils.struct_mask_parameters import (
    StructMaskParameters,
)

STRUCT_AXIS = [v.value for v in SupportedStructAxis]


@pytest.mark.parametrize("span", [3, 5])
@pytest.mark.parametrize("axis", [0, 1, 2] + STRUCT_AXIS)
def test_constructor(span, axis):
    """Test that the constructor correctly sets the attributes."""
    expected_axis = axis if isinstance(axis, int) else STRUCT_AXIS.index(axis)

    params = StructMaskParameters(axis=axis, span=span)
    assert params.axis == expected_axis

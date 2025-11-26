import pytest

from careamics.dataset_ng.image_stack.czi_image_stack import (
    are_czi_axes_valid,
)


@pytest.mark.parametrize(
    "axes, expected",
    [
        ("SCZYX", True),
        ("SCTYX", True),
        ("SCYX", True),
        ("CTYX", False),  # missing S axis
        ("SCZ", False),  # missing YX axes
        ("SCZYXT", False),  # extra axis
        ("TCYX", False),  # wrong order
    ],
)
def test_are_axes_valid(axes: str, expected: bool):
    """Test `are_axes_valid` function."""
    assert are_czi_axes_valid(axes) == expected

import numpy as np
import pytest

from careamics.utils import add_axes, check_axes_validity


@pytest.mark.parametrize(
    "axes, valid",
    [
        # Passing
        ("yx", True),
        ("Yx", True),
        ("Zyx", True),
        ("TzYX", True),
        ("SZYX", True),
        # Failing due to order
        ("XY", False),
        ("YXZ", False),
        ("YXT", False),
        ("ZTYX", False),
        # too few axes
        ("", False),
        ("X", False),
        # too many axes
        ("STZYX", False),
        # no yx axes
        ("ZT", False),
        ("ZY", False),
        # unsupported axes or axes pair
        ("STYX", False),
        ("CYX", False),
        # repeating characters
        ("YYX", False),
        ("YXY", False),
        # invalid characters
        ("YXm", False),
        ("1YX", False),
    ],
)
def test_are_axes_valid(axes, valid):
    """Test if axes are valid"""
    if valid:
        check_axes_validity(axes)
    else:
        with pytest.raises((ValueError, NotImplementedError)):
            check_axes_validity(axes)


@pytest.mark.parametrize(
    "axes, input_shape, output_shape",
    [
        ("YX", (8, 8), (1, 1, 8, 8)),
        ("YX", (1, 1, 8, 8), (1, 1, 8, 8)),
        ("ZYX", (8, 8, 8), (1, 1, 8, 8, 8)),
        ("ZYX", (1, 1, 8, 8, 8), (1, 1, 8, 8, 8)),
        ("SYX", (2, 8, 8), (2, 1, 8, 8)),
        ("SYX", (2, 1, 8, 8), (2, 1, 8, 8)),
        ("SZYX", (2, 8, 8, 8), (2, 1, 8, 8, 8)),
        ("SZYX", (2, 1, 8, 8, 8), (2, 1, 8, 8, 8)),
        ("TZYX", (2, 8, 8, 8), (2, 1, 8, 8, 8)),
        ("TZYX", (2, 1, 8, 8, 8), (2, 1, 8, 8, 8)),
    ],
)
def test_add_axes(axes, input_shape, output_shape):
    """Test that axes are added correctly."""
    input_array = np.zeros(input_shape)
    output = add_axes(input_array, axes)

    # check shape
    assert output.shape == output_shape

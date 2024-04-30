import pytest

from careamics.utils import check_axes_validity


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

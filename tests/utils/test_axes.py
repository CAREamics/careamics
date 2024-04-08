import pytest

from careamics.utils import check_axes_validity


@pytest.mark.parametrize(
    "axes, valid",
    [
        # Passing
        ("yx", True),
        ("Yx", True),
        ("Zyx", True),
        ("STYX", True),
        ("CYX", True),
        ("YXC", True),
        ("TzYX", True),
        ("SZYX", True),
        ("STZYX", True),
        ("XY", True),
        ("YXT", True),
        ("ZTYX", True),
        # non consecutive XY
        ("YZX", False),
        ("YZCXT", False),
        # too few axes
        ("", False),
        ("X", False),
        # no yx axes
        ("ZT", False),
        ("ZY", False),
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

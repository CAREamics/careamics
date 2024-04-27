import pytest

from careamics.config.validators import (
    check_axes_validity,
    patch_size_ge_than_8_power_of_2,
)


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


@pytest.mark.parametrize(
    "patch_size, error",
    [
        ((2, 8, 8), True),
        ((10,), True),
        ((8, 10, 16), True),
        ((8, 13), True),
        ((8, 16, 4), True),
        ((8,), False),
        ((8, 8), False),
        ((8, 64, 64), False),
    ],
)
def test_patch_size(patch_size, error):
    """Test if patch size is valid."""
    if error:
        with pytest.raises(ValueError):
            patch_size_ge_than_8_power_of_2(patch_size)
    else:
        patch_size_ge_than_8_power_of_2(patch_size)

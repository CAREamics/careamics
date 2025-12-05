import pytest

from careamics.config.validators import (
    patch_size_ge_than_8_power_of_2,
)


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

import pytest

from careamics.config.data.patching_strategies._patched_config import is_squared_in_yx


@pytest.mark.parametrize(
    "patch_size, expected",
    [
        ([32, 32], True),
        ([16, 32, 32], True),
        ([32, 16], False),
        ([16, 32], False),
    ],
)
def test_is_squared_in_yx(patch_size, expected):
    assert is_squared_in_yx(patch_size) == expected

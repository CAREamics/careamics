import itertools
from contextlib import nullcontext

import pytest

from careamics.config.data.patching_strategies._patched_config import is_squared_in_yx


@pytest.mark.parametrize(
    "patch_size, expected_error",
    # valid
    list(itertools.product([[32, 32], [16, 32, 32]], [nullcontext(0)]))
    # invalid
    + list(
        itertools.product(
            [[32, 16], [16, 32], [16, 16, 32]],
            [pytest.raises(ValueError, match="Patch size must be squared")],
        ),
    ),
)
def test_is_squared_in_yx(patch_size, expected_error):
    with expected_error:
        is_squared_in_yx(patch_size)

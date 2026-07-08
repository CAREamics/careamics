import itertools
from contextlib import nullcontext as does_not_raise

import pytest

from careamics.config.factories.care_n2n_factory import _validate_channel_dim

# test early stopping for care
# channel for both

# if input_channels>1 or channels is not none, need C in input axes

# if C in axes, channels need to be specified
# if output channel not specified, assume equal to input channel
# if target axes not specified, assume equal to input axes
# if C not in axes but output channel specified, need target axes


# -- Utility function


def expected_outcome(axes, target_axes, channels, in_channels, out_channels):
    # target_is_none = target_axes is None
    # target_has_c = not target_is_none and "C" in target_axes
    # axes_has_c = "C" in axes
    # channels_is_none = channels is None
    # out_larger_1 = out_channels is not None and out_channels > 1
    # in_larger_1 = in_channels is not None and in_channels > 1

    # if out_larger_1 and (target_is_none or not target_has_c):
    #     # must specify target axes with C
    #     return pytest.raises(ValueError)

    return does_not_raise(0)


# -- List of test parameters

AXES = ["YX", "CYX"]

TARGET_AXES = ["YX", "CYX"]

CHANNELS = [None, [1], [0, 2]]

IN_CHANNELS = [None, 1, 3]

OUT_CHANNELS = IN_CHANNELS

# -- Unit tests


@pytest.mark.parametrize(
    "axes, target_axes, channels, in_channels, out_channels",
    list(itertools.product(AXES, TARGET_AXES, CHANNELS, IN_CHANNELS, OUT_CHANNELS)),
)
def test_validate_channel_dim(axes, target_axes, channels, in_channels, out_channels):
    """Test channel dimension validation.

    While the utility function reimplements almost exactly the logic of the validation
    function, this test is useful because it covers all possible values and can catch
    errors being thrown that are not expected (and therefore not caught by the utility
    function).
    """
    exp_outcome = expected_outcome(
        axes, target_axes, channels, in_channels, out_channels
    )

    with exp_outcome:
        n_in, n_out = _validate_channel_dim(
            axes, target_axes, channels, in_channels, out_channels
        )

        # test model input/output channel size
        assert n_in == in_channels if in_channels is not None else 1
        if out_channels is None:
            assert n_out == in_channels if in_channels is not None else 1
        else:
            assert n_out == out_channels

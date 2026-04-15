import itertools
from contextlib import nullcontext

import pytest

from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
    get_unknown_parameters,
)
from careamics.config.support.supported_optimizers import (
    SupportedOptimizer,
    SupportedScheduler,
)

# SGD requires `lr`, the others don't have mandatory parameters
OPT = [opt.value for opt in SupportedOptimizer]
SGD = [SupportedOptimizer.SGD.value]

#
SCH_NO_STEP = [
    sch.value for sch in SupportedScheduler if sch != SupportedScheduler.STEP_LR
]
STEP = [SupportedScheduler.STEP_LR.value]

#


def a_test_func(param_1: str, param_2: int, param_3: float = 0.1):
    """Test function for _get_unknown_parameters."""
    pass


# ------------------------ Unit tests --------------------------


def test_get_unknown_parameters():
    """Test that _get_unknown_parameters returns the correct unknown parameters."""
    user_params = {
        "param_1": "test",
        "param_3": 0.2,
        "param_4": "unknown",
    }
    unknown_params = get_unknown_parameters(a_test_func, user_params)
    assert unknown_params == {"param_4": "unknown"}


@pytest.mark.parametrize(
    "opt_name, parameters, exp_error",
    # valid parameters
    list(itertools.product(OPT, [{"lr": 0.1}], [nullcontext(0)]))
    # additional unknown parameter
    + list(
        itertools.product(
            OPT,
            [{"lr": 0.1, "unknown_param": 42}],
            [pytest.raises(ValueError, match="Unknown parameters")],
        )
    )
    # missing `lr`` for SGD
    + list(
        itertools.product(
            SGD, [{}], [pytest.raises(ValueError, match="SGD optimizer requires")]
        )
    ),
)
def test_optimizer(opt_name, parameters, exp_error):
    """Test optimizer parameters filtering.

    For parameters, see:
    https://pytorch.org/docs/stable/optim.html#algorithms
    """
    with exp_error:
        OptimizerConfig(name=opt_name, parameters=parameters)


@pytest.mark.parametrize(
    "sch_name, parameters, exp_error",
    # valid parameters
    list(itertools.product(SCH_NO_STEP, [{}], [nullcontext(0)]))
    + list(itertools.product(STEP, [{"step_size": 10}], [nullcontext(0)]))
    # additional unknown parameter
    + list(
        itertools.product(
            SCH_NO_STEP,
            [{"unknown_param": 42}],
            [pytest.raises(ValueError, match="Unknown parameters")],
        )
    )
    + list(
        itertools.product(
            STEP,
            [{"step_size": 10, "unknown_param": 42}],
            [pytest.raises(ValueError, match="Unknown parameters")],
        )
    ),
)
def test_scheduler(sch_name, parameters, exp_error):
    """Test scheduler parameters filtering.

    For parameters, see:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    with exp_error:
        LrSchedulerConfig(name=sch_name, parameters=parameters)

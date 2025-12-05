from careamics.config import UNetBasedAlgorithm
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedLoss,
)


def test_all_algorithms_are_supported():
    """Test that all algorithms defined in the Literal are supported."""
    # list of supported algorithms
    algorithms = list(SupportedAlgorithm)

    # Algorithm json schema to extract the literal value
    schema = UNetBasedAlgorithm.model_json_schema()

    # check that all algorithms are supported
    for algo in schema["properties"]["algorithm"]["enum"]:
        assert algo in algorithms


# TODO: this should not support musplit and denoisplit losses
def test_all_losses_are_supported():
    """Test that all losses defined in the Literal are supported."""
    # list of supported losses
    losses = list(SupportedLoss)

    # Algorithm json schema
    schema = UNetBasedAlgorithm.model_json_schema()

    # check that all losses are supported
    for loss in schema["properties"]["loss"]["enum"]:
        assert loss in losses

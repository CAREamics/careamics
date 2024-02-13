import pytest

from careamics.config.algorithm_model import AlgorithmModel
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
)


def test_all_algorithms_are_supported():
    """Test that all algorithms defined in the Literal are supported."""
    # list of supported algorithms
    algorithms = list(SupportedAlgorithm)

    # Algorithm json schema to extract the literal value
    schema = AlgorithmModel.model_json_schema()

    # check that all algorithms are supported
    for algo in schema["properties"]["algorithm"]["enum"]:
        assert algo in algorithms


def test_supported_losses(minimum_algorithm):
    """Test that all supported losses are accepted by the AlgorithmModel."""
    for loss in SupportedLoss:
        minimum_algorithm["loss"] = loss.value
        AlgorithmModel(**minimum_algorithm)


def test_all_losses_are_supported():
    """Test that all losses defined in the Literal are supported."""
    # list of supported losses
    losses = list(SupportedLoss)

    # Algorithm json schema
    schema = AlgorithmModel.model_json_schema()

    # check that all losses are supported
    for loss in schema["properties"]["loss"]["enum"]:
        assert loss in losses


def test_model_discriminator(minimum_algorithm):
    """Test that discriminator permits correct assignment."""
    for model_name in SupportedArchitecture:
        # TODO change once VAE are implemented
        if model_name.value == "UNet":
            minimum_algorithm["model"]["architecture"] = model_name.value

            algo = AlgorithmModel(**minimum_algorithm)
            assert algo.model.architecture == model_name.value


@pytest.mark.parametrize(
    "algorithm, loss, model",
    [
        ("n2v", "n2v", {"architecture": "UNet", "n2v2": False}),
        ("n2v2", "n2v", {"architecture": "UNet", "n2v2": True}),
        ("structn2v", "n2v", {"architecture": "UNet", "n2v2": False}),
        ("custom", "mae", {"architecture": "UNet", "n2v2": True}),
    ],
)
def test_algorithm_constraints(algorithm: str, loss: str, model: dict):
    """Test that constraints are passed for each algorithm."""
    AlgorithmModel(algorithm=algorithm, loss=loss, model=model)

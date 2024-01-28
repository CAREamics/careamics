import pytest

from careamics.config.algorithm import AlgorithmModel
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
)


# from careamics.config.noise_models import NoiseModel


# def test_algorithm_noise_model():
#     d = {
#         "model_type": "hist",
#         "parameters": {"min_value": 324, "max_value": 3465},
#     }
#     NoiseModel(**d)


def test_supported_algorithm(minimum_algorithm):
    """Test that all supported algorithms are accepted by the AlgorithmModel."""
    for algo in SupportedAlgorithm:
        minimum_algorithm["algorithm"] = algo.value
        AlgorithmModel(**minimum_algorithm)


def test_all_algorithms_are_supported():
    """Test that all algorithms defined in the Literal are supported."""
    # list of supported algorithms
    algorithms = [algo for algo in SupportedAlgorithm]

    # Algorithm json schema
    schema = AlgorithmModel.model_json_schema()

    # check that all algorithms are supported
    for algo in schema["properties"]["algorithm"]['enum']:
       assert algo in algorithms


def test_supported_losses(minimum_algorithm):
    """Test that all supported losses are accepted by the AlgorithmModel."""
    for loss in SupportedLoss:
        minimum_algorithm["loss"] = loss.value
        AlgorithmModel(**minimum_algorithm)


def test_all_losses_are_supported():
    """Test that all losses defined in the Literal are supported."""
    # list of supported losses
    losses = [loss for loss in SupportedLoss]

    # Algorithm json schema
    schema = AlgorithmModel.model_json_schema()

    # check that all losses are supported
    for loss in schema["properties"]["loss"]['enum']:
       assert loss in losses


def test_model_discriminator(minimum_algorithm):
    """Test that discriminator permits correct assignment."""
    for model_name in SupportedArchitecture:
    
        minimum_algorithm["model"]["architecture"] = model_name.value

        algo = AlgorithmModel(**minimum_algorithm)
        assert algo.model.architecture == model_name


def test_wrong_values_by_assigment(minimum_algorithm: dict):
    """Test that wrong values are not accepted through assignment."""
    algo = AlgorithmModel(**minimum_algorithm)

    # loss
    with pytest.raises(ValueError):
        algo.loss = "ms-meh"
    assert algo.loss == minimum_algorithm["loss"]

    # model
    with pytest.raises(ValueError):
        algo.model.architecture = "YouNet"

    # optimizer
    with pytest.raises(ValueError):
        algo.optimizer = "I'd rather not to."

    # lr_scheduler
    with pytest.raises(ValueError):
        algo.lr_scheduler = "Why don't YOU schedule it for once?"

    

def test_model_dump(minimum_algorithm: dict):
    """Test that default values are excluded from model dump with
    `exclude_defaults=True`."""
    algo = AlgorithmModel(**minimum_algorithm)

    # dump model
    model_dict = algo.model_dump(exclude_defaults=True)

    # check that default values are excluded except the architecture
    assert len(model_dict) == 5
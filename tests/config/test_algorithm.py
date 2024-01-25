import pytest

from careamics.config.algorithm import Algorithm
from careamics.config.torch_optim import (
    OptimizerModel,
    LrSchedulerModel,
    TorchOptimizer,
    TorchLRScheduler,
)


# from careamics.config.noise_models import NoiseModel


# def test_algorithm_noise_model():
#     d = {
#         "model_type": "hist",
#         "parameters": {"min_value": 324, "max_value": 3465},
#     }
#     NoiseModel(**d)


@pytest.mark.parametrize("model_name", ["VAE", "UNet"])
def test_model_discriminator(model_name):
    """Test that discriminator permits correct assignment."""
    architecture_config = {
        "algorithm_type": "n2v",
        "loss": "n2v",
        "model": {"architecture": model_name},
        "optimizer": {"name": "Adam"},
        "lr_scheduler": {"name": "ReduceLROnPlateau"},
    }

    algo = Algorithm(**architecture_config)
    assert algo.model.architecture == model_name


def test_wrong_values_by_assigment(minimum_algorithm: dict):
    """Test that wrong values are not accepted through assignment."""
    algo = Algorithm(**minimum_algorithm)

    # loss
    algo.loss = minimum_algorithm["loss"]
    with pytest.raises(ValueError):
        algo.loss = "ms-meh"
    assert algo.loss == minimum_algorithm["loss"]

    # model
    algo.model = minimum_algorithm["model"]
    with pytest.raises(ValueError):
        algo.model.architecture = "YouNet"

    # optimizer
    algo.optimizer = OptimizerModel(name=TorchOptimizer.Adam, parameters={"lr": 0.1})
    with pytest.raises(ValueError):
        algo.optimizer = "I'd rather not to."

    # lr_scheduler
    algo.lr_scheduler = LrSchedulerModel(
        name=TorchLRScheduler.ReduceLROnPlateau, parameters={"factor": 0.1}
    )
    with pytest.raises(ValueError):
        algo.lr_scheduler = "Why don't you schedule it for once? :)"

    
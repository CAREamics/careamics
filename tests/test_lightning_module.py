from careamics import CAREamicsModule
from careamics.config import AlgorithmModel


def test_careamics_module(minimum_algorithm):
    """Test CAREamicsModule class as an intermediate layer."""
    algo_config = AlgorithmModel(**minimum_algorithm)

    # extract model parameters
    model_parameters = algo_config.model.model_dump(exclude_none=True)
    model_parameters.pop("architecture")

    # instantiate CAREamicsModule
    CAREamicsModule(
        algorithm=algo_config.algorithm,
        loss=algo_config.loss,
        architecture=algo_config.model.architecture,
        model_parameters=model_parameters,
        optimizer=algo_config.optimizer.name,
        optimizer_parameters=algo_config.optimizer.parameters,
        lr_scheduler=algo_config.lr_scheduler.name,
        lr_scheduler_parameters=algo_config.lr_scheduler.parameters,
    )

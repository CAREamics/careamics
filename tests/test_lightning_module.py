from careamics.config import AlgorithmModel
from careamics.lightning_module import CAREamicsKiln, CAREamicsModule


def test_careamics_module(minimum_algorithm_n2v):
    """Test that the minimum algorithm allows instantiating a the Lightning API
    intermediate layer."""
    algo_config = AlgorithmModel(**minimum_algorithm_n2v)

    # extract model parameters
    model_parameters = algo_config.model.model_dump(exclude_none=True)

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


def test_careamics_kiln(minimum_algorithm_n2v):
    """Test that the minimum algorithm allows instantiating a CAREamicsKiln."""
    algo_config = AlgorithmModel(**minimum_algorithm_n2v)

    # instantiate CAREamicsKiln
    CAREamicsKiln(algo_config)

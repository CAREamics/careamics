from careamics.losses.loss_factory import loss_factory, loss_parameters_factory
from careamics.losses.lvae.losses import denoisplit_loss, musplit_loss
import pytest
from careamics.config import AlgorithmConfig
from careamics.lightning_module import CAREamicsModule, CAREamicsModuleWrapper


def test_mu_split_loss(minimum_algorithm_musplit):
    loss = loss_factory("musplit")
    assert loss == musplit_loss


    algo_config = AlgorithmConfig(**minimum_algorithm_musplit)

    # extract model parameters
    model_parameters = algo_config.model.model_dump(exclude_none=True)

    # define default loss parameters
    loss_parameters = loss_parameters_factory("musplit")

    # instantiate CAREamicsModule
    CAREamicsModuleWrapper(
        algorithm=algo_config.algorithm,
        loss=algo_config.loss,
        architecture=algo_config.model.architecture,
        model_parameters=model_parameters,
        optimizer=algo_config.optimizer.name,
        optimizer_parameters=algo_config.optimizer.parameters,
        lr_scheduler=algo_config.lr_scheduler.name,
        lr_scheduler_parameters=algo_config.lr_scheduler.parameters,
    )
    

def test_denoisplit_loss():
    loss = loss_factory("denoisplit")
    assert loss == denoisplit_loss

    loss_parameters = loss_parameters_factory("denoisplit")
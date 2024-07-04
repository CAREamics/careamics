from careamics.config import AlgorithmConfig
from careamics.lightning_module import CAREamicsModule
from careamics.losses.loss_factory import loss_factory, loss_parameters_factory
from careamics.losses.lvae.losses import denoisplit_loss, musplit_loss
import torch


def test_mu_split_loss(minimum_algorithm_musplit):
    loss_func = loss_factory("musplit")
    assert loss_func == musplit_loss

    algo_config = AlgorithmConfig(**minimum_algorithm_musplit)

    # instantiate CAREamicsModule
    module = CAREamicsModule(
        algorithm_config=algo_config
    )
    inputs = torch.rand(2, 2, 5, 64, 64)
    predictions = torch.rand(1, 1, 8, 8)
    module.training_step(inputs, 0)
    loss_value = loss_func(predictions, module)


def test_denoisplit_loss():
    loss = loss_factory("denoisplit")
    assert loss == denoisplit_loss

    loss_parameters = loss_parameters_factory("denoisplit")

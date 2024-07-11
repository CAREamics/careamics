import torch

from careamics.config import AlgorithmConfig
from careamics.lightning.lightning_module import VAEModule
from careamics.losses.loss_factory import loss_factory
from careamics.losses.lvae.losses import denoisplit_loss, musplit_loss


def test_mu_split_loss(minimum_algorithm_musplit):
    loss_func = loss_factory("musplit")
    assert loss_func == musplit_loss

    algo_config = AlgorithmConfig(**minimum_algorithm_musplit)

    # instantiate CAREamicsModule
    module = VAEModule(algorithm_config=algo_config)
    inputs = torch.rand(2, 2, 5, 64, 64)
    step = module.training_step(inputs, 0)
    for k in step:
        assert not step[k].isnan()


def test_denoisplit_loss(minimum_algorithm_denoisplit):
    loss = loss_factory("denoisplit")
    assert loss == denoisplit_loss

    algo_config = AlgorithmConfig(**minimum_algorithm_denoisplit)

    # instantiate CAREamicsModule
    module = VAEModule(algorithm_config=algo_config)
    inputs = torch.rand(2, 2, 5, 64, 64)
    step = module.training_step(inputs, 0)
    for k in step:
        assert not step[k].isnan()

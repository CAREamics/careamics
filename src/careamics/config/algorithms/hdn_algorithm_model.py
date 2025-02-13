from typing import Literal

from careamics.config.algorithms.vae_algorithm_model import VAEBasedAlgorithm
from careamics.config.architectures import LVAEModel
from careamics.config.loss_model import LVAELossConfig


class HDNAlgorithm(VAEBasedAlgorithm):
    algorithm: Literal["hdn"] = "hdn"

    loss: LVAELossConfig

    model: LVAEModel  # TODO add validators

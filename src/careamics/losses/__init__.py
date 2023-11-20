"""Losses module."""


from .loss_factory import create_loss_function as create_loss_function
from .loss_factory import create_noise_model as create_noise_model
from .noise_models import GaussianMixtureNoiseModel as GaussianMixtureNoiseModel
from .noise_models import HistogramNoiseModel as HistogramNoiseModel

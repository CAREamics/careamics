"""Tests for the pytest fixtures."""

from careamics.config import Configuration, N2VConfiguration
from careamics.config.algorithms import UNetBasedAlgorithm
from careamics.config.data import DataConfig, N2VDataConfig
from careamics.config.inference_model import InferenceConfig
from careamics.config.training_model import TrainingConfig


def test_minimum_algorithm(minimum_algorithm_n2v):
    # create algorithm configuration
    UNetBasedAlgorithm(**minimum_algorithm_n2v)


def test_minimum_data(minimum_data):
    # create data configuration
    DataConfig(**minimum_data)


def test_minimum_prediction(minimum_inference):
    # create prediction configuration
    InferenceConfig(**minimum_inference)


def test_minimum_training(minimum_training):
    # create training configuration
    TrainingConfig(**minimum_training)


def test_minimum_data_n2v(minimum_data_n2v):
    # create data configuration
    N2VDataConfig(**minimum_data_n2v)


def test_minimum_n2v_configuration(minimum_n2v_configuration):
    # create configuration
    N2VConfiguration(**minimum_n2v_configuration)


def test_minimum_configuration(minimum_supervised_configuration):
    # create configuration
    Configuration(**minimum_supervised_configuration)

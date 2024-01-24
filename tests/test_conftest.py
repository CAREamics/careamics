from careamics import Configuration
from careamics.config.algorithm import Algorithm
from careamics.config.data import Data
from careamics.config.training import Training


def test_minimum_algorithm(minimum_algorithm):
    # create algorithm configuration
    Algorithm(**minimum_algorithm)


def test_minimum_data(minimum_data):
    # create data configuration
    Data(**minimum_data)


def test_minimum_training(minimum_training):
    # create training configuration
    Training(**minimum_training)


def test_minimum_configuration(minimum_configuration):
    # create configuration
    Configuration(**minimum_configuration)


def test_complete_configuration(complete_configuration):
    # create configuration
    Configuration(**complete_configuration)
from careamics import Configuration
from careamics.config.algorithm_model import AlgorithmModel
from careamics.config.data_model import DataModel
from careamics.config.training_model import Training


def test_minimum_algorithm(minimum_algorithm):
    # create algorithm configuration
    AlgorithmModel(**minimum_algorithm)


def test_minimum_data(minimum_data):
    # create data configuration
    DataModel(**minimum_data)


def test_minimum_training(minimum_training):
    # create training configuration
    Training(**minimum_training)


def test_minimum_configuration(minimum_configuration):
    # create configuration
    Configuration(**minimum_configuration)
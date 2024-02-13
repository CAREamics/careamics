import pytest

from careamics import CAREamist, Configuration, save_configuration


def test_no_parameters():
    """Test that CAREamics cannot be instantiated without parameters."""
    with pytest.raises(ValueError):
        CAREamist()


def test_minimum_configuration_via_object(minimum_configuration):
    """Test that CAREamics can be instantiated with a minimum configuration object."""
    # create configuration
    config = Configuration(**minimum_configuration)

    # instantiate CAREamist
    CAREamist(configuration=config)


def test_minimum_configuration_via_path(tmp_path, minimum_configuration):
    """Test that CAREamics can be instantiated with a path to a minimum
    configuration.
    """
    # create configuration
    config = Configuration(**minimum_configuration)
    path_to_config = save_configuration(config, tmp_path)

    # instantiate CAREamist
    CAREamist(path_to_config=path_to_config)

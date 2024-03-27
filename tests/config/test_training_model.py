import pytest

from careamics.config.training_model import AMP, TrainingModel


@pytest.mark.parametrize("init_scale", [512, 1024, 65536])
def test_amp_init_scale(init_scale: int):
    """Test AMP init_scale parameter."""
    amp = AMP(use=True, init_scale=init_scale)
    assert amp.init_scale == init_scale


@pytest.mark.parametrize("init_scale", [511, 1088, 65537])
def test_amp_wrong_init_scale(init_scale: int):
    """Test wrong AMP init_scale parameter."""
    with pytest.raises(ValueError):
        AMP(use=True, init_scale=init_scale)


def test_amp_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    amp = AMP(use=True, init_scale=1024)

    # use
    amp.use = False
    with pytest.raises(ValueError):
        amp.use = None

    with pytest.raises(ValueError):
        amp.use = 3

    # init_scale
    amp.init_scale = 512
    with pytest.raises(ValueError):
        amp.init_scale = "1026"


def test_training_wrong_values_by_assignments(minimum_training: dict):
    """Test that wrong values cause an error during assignment."""
    training = TrainingModel(**minimum_training)

    # num_epochs
    training.num_epochs = 2
    with pytest.raises(ValueError):
        training.num_epochs = -1

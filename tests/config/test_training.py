import pytest
from pydantic import conlist


from careamics.config.training import AMP, Training


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
    training = Training(**minimum_training)

    # num_epochs
    training.num_epochs = 2
    with pytest.raises(ValueError):
        training.num_epochs = -1

    # batch_size
    training.batch_size = 2
    with pytest.raises(ValueError):
        training.batch_size = -1

    # use_wandb
    training.use_wandb = True
    with pytest.raises(ValueError):
        training.use_wandb = None

    # amp
    training.amp = AMP(use=True, init_scale=1024)
    with pytest.raises(ValueError):
        training.amp = "I don't want to use AMP."

    # num_workers
    training.num_workers = 2
    with pytest.raises(ValueError):
        training.num_workers = -1

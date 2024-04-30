import pytest

from careamics.config.training_model import TrainingModel


def test_training_wrong_values_by_assignments(minimum_training: dict):
    """Test that wrong values cause an error during assignment."""
    training = TrainingModel(**minimum_training)

    # num_epochs
    training.num_epochs = 2
    with pytest.raises(ValueError):
        training.num_epochs = -1

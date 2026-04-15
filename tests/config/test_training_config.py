import pytest
from pydantic import ValidationError

from careamics.config.lightning.training_config import TrainingConfig
from careamics.config.ng_configs.ng_training_configuration import NGTrainingConfig


# tests for old careamist
def test_training_config_default_logger_is_none():
    """Test that the default logger value is None."""
    config = TrainingConfig()
    assert config.logger is None


@pytest.mark.parametrize("logger", ["wandb", "tensorboard"])
def test_training_config_valid_logger(logger: str):
    """Test that TrainingConfig accepts 'wandb' and 'tensorboard'."""
    config = TrainingConfig(logger=logger)
    assert config.logger == logger


@pytest.mark.parametrize("logger", ["mlflow", "WandB", "TensorBoard", "", "csv"])
def test_training_config_invalid_logger(logger: str):
    """Test that TrainingConfig rejects unsupported or incorrectly-cased
    logger strings.
    """
    with pytest.raises(ValidationError):
        TrainingConfig(logger=logger)


def test_training_config_has_logger_false_by_default():
    """Test has_logger returns False when no logger is set."""
    config = TrainingConfig()
    assert not config.has_logger()


@pytest.mark.parametrize("logger", ["wandb", "tensorboard"])
def test_training_config_has_logger_true(logger: str):
    """Test has_logger returns True when a valid logger is set."""
    config = TrainingConfig(logger=logger)
    assert config.has_logger()


# tests for careamist_v2
def test_ng_training_config_default_logger_is_none():
    """Test that the default logger value is None."""
    config = NGTrainingConfig()
    assert config.logger is None


@pytest.mark.parametrize("logger", ["wandb", "tensorboard"])
def test_ng_training_config_valid_logger(logger: str):
    """Test NGTrainingConfig accepts 'wandb' and 'tensorboard'."""
    config = NGTrainingConfig(logger=logger)
    assert config.logger == logger


@pytest.mark.parametrize("logger", ["mlflow", "WandB", "TensorBoard", "", "csv"])
def test_ng_training_config_invalid_logger(logger: str):
    """Test that NGTrainingConfig rejects unsupported or incorrectly-cased
    logger strings.
    """
    with pytest.raises(ValidationError):
        NGTrainingConfig(logger=logger)

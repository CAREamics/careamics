from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.config.lightning.training_configuration import (
    SelfSupervisedCheckpointing,
    SupervisedCheckpointing,
    TrainingConfig,
    default_training_dict,
    default_training_factory,
)


@pytest.mark.parametrize(
    "preset", [SupervisedCheckpointing(), SelfSupervisedCheckpointing()]
)
def test_checkpoint_presets(preset):
    """Tests that the presets are valid parametrization for ModelCheckpoint."""
    ModelCheckpoint(**asdict(preset))


@pytest.mark.parametrize("algo", ["care", "n2n", "n2v"])
@pytest.mark.parametrize(
    "checkpoint_params",
    [
        None,
        {"save_top_k": 5},
    ],
)
def test_default_training_dict(algo, checkpoint_params):
    """Tests that the default training configuration can be created."""
    config = default_training_dict(
        algorithm=algo,
        checkpoint_params=checkpoint_params,
    )
    TrainingConfig(**config)


@pytest.mark.parametrize("algo", ["care", "n2n", "n2v"])
def test_default_training_factory(algo):
    """Tests that the default training factory can be created."""
    validated_dict = {
        "algorithm_config": SimpleNamespace(
            algorithm=algo,
            monitor_metric="train_loss",
        )
    }

    config = default_training_factory(validated_dict)
    assert isinstance(config, TrainingConfig)

    # check the algorithm defaults
    if algo == "care":
        assert config.checkpoint_params == asdict(SupervisedCheckpointing())
        assert config.early_stopping_params is not None
    else:
        assert config.checkpoint_params == asdict(SelfSupervisedCheckpointing())
        assert config.early_stopping_params is None

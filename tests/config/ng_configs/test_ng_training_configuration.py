from dataclasses import asdict

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.config.ng_configs.ng_training_configuration import (
    SelfSupervisedCheckpointing,
    SupervisedCheckpointing,
)


@pytest.mark.parametrize(
    "preset", [SupervisedCheckpointing(), SelfSupervisedCheckpointing()]
)
def test_checkpoint_presets(preset):
    """Tests that the presets are valid parametrization for ModelCheckpoint."""
    ModelCheckpoint(**asdict(preset))

"""Test StopPredictionCallback."""

import pytest
from pytorch_lightning import Trainer

from careamics.lightning.callbacks import (
    PredictionStoppedException,
    StopPredictionCallback,
)










def test_callback_with_stateful_condition():
    """Test callback responds to changing stop condition."""
    stop_flag = {"value": False}
    callback = StopPredictionCallback(stop_condition=lambda: stop_flag["value"])
    trainer = Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)

    callback.on_predict_batch_start(
        trainer=trainer, pl_module=None, batch=None, batch_idx=0
    )
    assert not trainer.should_stop

    stop_flag["value"] = True

    with pytest.raises(PredictionStoppedException):
        callback.on_predict_batch_start(
            trainer=trainer, pl_module=None, batch=None, batch_idx=1
        )

    assert trainer.should_stop

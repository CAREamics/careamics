"""Test StopPredictionCallback."""

import pytest
from pytorch_lightning import Trainer

from careamics.lightning.callbacks import (
    PredictionStoppedException,
    StopPredictionCallback,
)




def test_callback_initialization():
    """Test callback can be initialized with stop condition."""
    callback = StopPredictionCallback(stop_condition=lambda: False)
    assert callback is not None
    assert callable(callback.stop_condition)


def test_callback_continues_when_condition_false():
    """Test prediction continues when stop condition is False."""
    callback = StopPredictionCallback(stop_condition=lambda: False)
    trainer = Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)

    callback.on_predict_batch_start(
        trainer=trainer,
        pl_module=None,
        batch=None,
        batch_idx=0,
        dataloader_idx=0,
    )

    assert not trainer.should_stop


def test_callback_stops_when_condition_true():
    """Test prediction stops when stop condition is True."""
    callback = StopPredictionCallback(stop_condition=lambda: True)
    trainer = Trainer(fast_dev_run=True, enable_checkpointing=False, logger=False)

    with pytest.raises(PredictionStoppedException):
        callback.on_predict_batch_start(
            trainer=trainer,
            pl_module=None,
            batch=None,
            batch_idx=0,
            dataloader_idx=0,
        )

    assert trainer.should_stop


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

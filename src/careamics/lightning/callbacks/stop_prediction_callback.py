"""Callback for stopping prediction based on external condition."""

from collections.abc import Callable
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class PredictionStoppedException(Exception):
    """Exception raised when prediction is stopped by external signal."""

    pass


class StopPredictionCallback(Callback):
    """PyTorch Lightning callback to stop prediction based on external condition.

    This callback monitors a user-provided stop condition at the start of each
    prediction batch. When the condition is met, the callback stops the trainer
    and raises PredictionStoppedException to interrupt the prediction loop.

    Parameters
    ----------
    stop_condition : Callable[[], bool]
        A callable that returns True when prediction should stop. The callable
        is invoked at the start of each prediction batch.
    """

    def __init__(self, stop_condition: Callable[[], bool]) -> None:
        """Initialize the callback with a stop condition.

        Parameters
        ----------
        stop_condition : Callable[[], bool]
            Function that returns True when prediction should stop.
        """
        super().__init__()
        self.stop_condition = stop_condition

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Check stop condition at the start of each prediction batch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer instance.
        pl_module : LightningModule
            Lightning module being used for prediction.
        batch : Any
            Current batch of data.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader, by default 0.

        Raises
        ------
        PredictionStoppedException
            If stop_condition() returns True.
        """
        if self.stop_condition():
            trainer.should_stop = True
            raise PredictionStoppedException("Prediction stopped by user")

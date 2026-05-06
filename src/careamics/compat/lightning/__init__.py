"""Deprecated Lightning modules from CAREamics v0.1.0."""

# Re-export ProgressBarCallback from the non-compat callbacks
from careamics.lightning.callbacks.progress_bar_callback import ProgressBarCallback

from .lightning_module import FCNModule, create_careamics_module
from .predict_data_module import PredictDataModule, create_predict_datamodule
from .train_data_module import TrainDataModule, create_train_datamodule

__all__ = [
    "FCNModule",
    "PredictDataModule",
    "ProgressBarCallback",
    "TrainDataModule",
    "create_careamics_module",
    "create_predict_datamodule",
    "create_train_datamodule",
]

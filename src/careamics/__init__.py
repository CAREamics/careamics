"""Main CAREamics module."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CAREamist",
    "CAREamicsModuleWrapper",
    "CAREamicsPredictData",
    "CAREamicsTrainData",
    "Configuration",
    "load_configuration",
    "save_configuration",
    "TrainingDataWrapper",
    "PredictDataWrapper",
]

from .careamist import CAREamist
from .config import Configuration, load_configuration, save_configuration
from .lightning_datamodule import CAREamicsTrainData, TrainingDataWrapper
from .lightning_module import CAREamicsModuleWrapper
from .lightning_prediction_datamodule import CAREamicsPredictData, PredictDataWrapper

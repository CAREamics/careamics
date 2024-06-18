from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer

import careamics

from ..config.tile_information import TileInformation
from .stitch_prediction import stitch_prediction


class PredictionManager:
    """
    Encapsulates logic for converting `pytorch-lightning.Trainer.predict` outputs.

    Cases include: combining
    """

    def __init__(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: "careamics.CAREamicsPredictData",
        # datamodule: LightningDataModule,
        checkpoint: Optional[Literal["best", "last"]] = None,
    ):
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        self.checkpoint = checkpoint

    def predict(self) -> Union[List[NDArray], NDArray]:
        predictions = self.trainer.predict(
            model=self.model, datamodule=self.datamodule, ckpt_path=self.checkpoint
        )
        if len(predictions) == 0:
            return predictions

        predictions = self.combine_batches(predictions)
        if self.datamodule.tiled:
            predictions = stitch_prediction(*predictions)

        # TODO: add this in? Returns output with same axes as input
        # Won't work with tiling rn because stitch_prediction func removes axes
        # predictions = self.reshape(predictions)

        # TODO: might want to remove this
        if (isinstance(predictions, list)) and (len(predictions) == 1):
            return predictions[0]
        return predictions

    def combine_batches(
        self, predictions: List[Any]
    ) -> Union[List[NDArray], Tuple[NDArray, List[TileInformation]]]:
        if self.datamodule.tiled:
            return self._combine_tiled_batches(predictions)
        else:
            return self._combine_untiled_batches(predictions)

    @staticmethod
    def _combine_tiled_batches(
        predictions: List[Tuple[NDArray, List[TileInformation]]]
    ) -> Tuple[NDArray, List[TileInformation]]:

        # turn list of lists into single list
        tile_infos = [
            tile_info
            for _, tile_info_list in predictions
            for tile_info in tile_info_list
        ]
        predictions = np.concatenate([preds for preds, _ in predictions])
        return predictions, tile_infos

    @staticmethod
    def _combine_untiled_batches(predictions: List[NDArray]) -> List[NDArray]:
        predictions = np.concatenate(predictions, axis=0)
        return np.split(predictions, predictions.shape[0], axis=0)

    def reshape(self, predictions: List[NDArray]) -> List[NDArray]:
        axes = self.datamodule.prediction_config.axes
        if "C" not in axes:
            predictions = [pred[:, 0] for pred in predictions]
        if "S" not in axes:
            predictions = [pred[0] for pred in predictions]
        return predictions

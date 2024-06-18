from typing import List, Any, Optional, Literal

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import Trainer, LightningModule

from careamics import CAREamicsPredictData
from .stitch_prediction import stitch_prediction

def combine_tiled_batches(predictions: List[Any]):
    tile_infos = [
        tile_info for _, tile_info_list in predictions for tile_info in tile_info_list
    ]
    predictions = np.concatenate([preds for preds, _ in predictions])
    return predictions, tile_infos


class PredictionManager:

    def __init__(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: CAREamicsPredictData,
        checkpoint: Optional[Literal["best", "last"]] = None,
    ): 
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        self.checkpoint = checkpoint

    def predict(self):
        predictions = self.trainer.predict(
            model=self.model, datamodule=self.datamodule, ckpt_path=self.checkpoint
        )
        if len(predictions) == 0:
            return predictions
        
        predictions = self.combine_batches(predictions)
        if self.datamodule.tiled:
            predictions = stitch_prediction(*predictions)

        # TODO: might want to remove this
        if (isinstance(predictions, list)) and (len(predictions) == 1):
            return predictions[0]
        return predictions

    def combine_batches(self, predictions: List[Any]):
        if self.datamodule.tiled:
            return self._combine_tiled_batches(predictions)
        else:
            return [np.concatenate(predictions, axis=0)]

    @staticmethod
    def _combine_tiled_batches(predictions: List[Any]):

        # turn list of lists into single lust
        tile_infos = [
            tile_info 
            for _, tile_info_list in predictions 
            for tile_info in tile_info_list
        ]
        predictions = np.concatenate([preds for preds, _ in predictions])
        return predictions, tile_infos
"""Module containing `PredictionManager` class."""

from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer

import careamics

from ..config.tile_information import TileInformation
from .stitch_prediction import stitch_prediction


class PredictionManager:
    """
    Encapsulate logic for converting `pytorch-lightning.Trainer.predict` outputs.

    Cases include: stiching tiled predictions - batched & not batched,
    combining batched predictions.

    Parameters
    ----------
    trainer : Trainer
        Trainer from `pytorch-lightning`.
    model : LightningModule
        Model that will output predictions.
    datamodule : careamics.CAREamicsPredictData
        Data as input to model.
    checkpoint : {"best", "last"}, optional
        Checkpoint path.

    Attributes
    ----------
    trainer : Trainer
        Trainer from `pytorch-lightning`.
    model : LightningModule
        Model that will output predictions.
    datamodule : careamics.CAREamicsPredictData
        Data as input to model.
    checkpoint : {"best", "last"}, optional
        Checkpoint path.
    """

    def __init__(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: "careamics.CAREamicsPredictData",
        checkpoint: Optional[Literal["best", "last"]] = None,
    ):
        """
        Encapsulate logic for converting `pytorch-lightning.Trainer.predict` outputs.

        Parameters
        ----------
        trainer : Trainer
            Trainer from `pytorch-lightning`.
        model : LightningModule
            Model that will output predictions.
        datamodule : careamics.CAREamicsPredictData
            Data as input to model.
        checkpoint : {"best", "last"}, optional
            Checkpoint path.
        """
        self.trainer = trainer
        self.model = model
        self.datamodule = datamodule
        self.checkpoint = checkpoint

    def predict(self) -> Union[List[NDArray], NDArray]:
        """
        Create predictions and converts the outputs to the desired form.

        Returns
        -------
        list of numpy.ndarray or numpy.ndarray
            List of arrays with the axes SC(Z)YX. If there is only 1 output it will not
            be in a list.
        """
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
        """
        If predictions are in batches, they will be combined.

        Parameters
        ----------
        predictions : list
            Predictions that are output from `Trainer.predict`.

        Returns
        -------
        (list of numpy.ndarray) or (tuple of (numpy.ndarray, list of TileInformation))
            Combined batches.
        """
        if self.datamodule.tiled:
            return self._combine_tiled_batches(predictions)
        else:
            return self._combine_untiled_batches(predictions)

    @staticmethod
    def _combine_tiled_batches(
        predictions: List[Tuple[NDArray, List[TileInformation]]]
    ) -> Tuple[NDArray, List[TileInformation]]:
        """
        Combine batches from tiled output.

        Parameters
        ----------
        predictions : list
            Predictions that are output from `Trainer.predict`.

        Returns
        -------
        tuple of (numpy.ndarray, list of TileInformation)
            Combined batches.
        """
        # turn list of lists into single list
        tile_infos = [
            tile_info
            for _, tile_info_list in predictions
            for tile_info in tile_info_list
        ]
        prediction_tiles: NDArray = np.concatenate([preds for preds, _ in predictions])
        return prediction_tiles, tile_infos

    @staticmethod
    def _combine_untiled_batches(predictions: List[NDArray]) -> List[NDArray]:
        """
        Combine batches from un-tiled output.

        Parameters
        ----------
         predictions : list
             Predictions that are output from `Trainer.predict`.

        Returns
        -------
         list of nunpy.ndarray
             Combined batches.
        """
        prediction_concat: NDArray = np.concatenate(predictions, axis=0)
        return np.split(prediction_concat, prediction_concat.shape[0], axis=0)

    def reshape(self, predictions: List[NDArray]) -> List[NDArray]:
        """
        Reshape predictions to have dimensions of input.

        Parameters
        ----------
        predictions : List[NDArray]
            Predictions after being processed `combine_batches` method.

        Returns
        -------
        List[NDArray]
            Reshaped predicitions.
        """
        axes = self.datamodule.prediction_config.axes
        if "C" not in axes:
            predictions = [pred[:, 0] for pred in predictions]
        if "S" not in axes:
            predictions = [pred[0] for pred in predictions]
        return predictions

from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loops.fetchers import _DataLoaderIterDataFetcher
from pytorch_lightning.loops.utilities import _no_grad_context
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT

from careamics.prediction import stitch_prediction
from careamics.utils import denormalize


class CAREamicsFiring(L.loops._PredictionLoop):
    """Predict loop for tiles-based prediction."""

    # def _predict_step(self, batch, batch_idx, dataloader_idx, dataloader_iter):
    #     self.model.predict_step(batch, batch_idx)

    def _on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` hook.

        Returns
        -------
            the results for all dataloaders

        """
        trainer = self.trainer
        call._call_callback_hooks(trainer, "on_predict_epoch_end")
        call._call_lightning_module_hook(trainer, "on_predict_epoch_end")

        if self.return_predictions:
            return self.predicted_array
        return None

    @_no_grad_context
    def run(self) -> Optional[_PREDICT_OUTPUT]:
        self.setup_data()
        if self.skip:
            return None
        self.reset()
        self.on_run_start()
        data_fetcher = self._data_fetcher
        assert data_fetcher is not None

        self.predicted_array = []
        self.tiles = []
        self.stitching_data = []

        while True:
            try:
                if isinstance(data_fetcher, _DataLoaderIterDataFetcher):
                    dataloader_iter = next(data_fetcher)
                    # hook's batch_idx and dataloader_idx arguments correctness cannot
                    # be guaranteed in this setting
                    batch = data_fetcher._batch
                    batch_idx = data_fetcher._batch_idx
                    dataloader_idx = data_fetcher._dataloader_idx
                else:
                    dataloader_iter = None
                    batch, batch_idx, dataloader_idx = next(data_fetcher)
                self.batch_progress.is_last_batch = data_fetcher.done
                # run step hooks
                self._predict_step(batch, batch_idx, dataloader_idx, dataloader_iter)

                # Stitching tiles together
                last_tile, *data = self.predictions[batch_idx][1]
                self.tiles.append(self.predictions[batch_idx][0])
                self.stitching_data.append(data)
                if last_tile:
                    predicted_sample = stitch_prediction(
                        self.tiles, self.stitching_data
                    )
                    # TODO replace with Albu class
                    denormalized_sample = denormalize(
                        predicted_sample,
                        self._data_source.instance.predict_dataset.mean,
                        self._data_source.instance.predict_dataset.std,
                    )
                    self.predicted_array.append(denormalized_sample)
                    self.tiles.clear()
                    self.stitching_data.clear()
            except StopIteration:
                break
            finally:
                self._restarting = False
        return self.on_run_end()


def predict_tiled_simple(
    predictions: list,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Predict using tiling.

    Parameters
    ----------
    pred_loader : DataLoader
        Prediction dataloader.
    progress_bar : ProgressBar
        Progress bar.
    tta : bool, optional
        Whether to use test time augmentation, by default True.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
        Predicted image, or list of predictions if the images have different sizes.

    Warns
    -----
    UserWarning
        If the samples have different shapes, the prediction then returns a list.
    """
    prediction = []
    tiles = []
    stitching_data = []

    for _i, (_tile, *auxillary) in enumerate(predictions):
        # Unpack auxillary data into last tile indicator and data, required to
        # stitch tiles together
        if auxillary:
            last_tile, *stitching_data = auxillary

        if last_tile:
            # Stitch tiles together if sample is finished
            predicted_sample = stitch_prediction(tiles, stitching_data)
            prediction.append(predicted_sample)
            tiles.clear()
            stitching_data.clear()

        try:
            return np.stack(prediction)
        except ValueError:
            return prediction

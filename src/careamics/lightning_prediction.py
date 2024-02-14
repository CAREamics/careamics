from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loops.fetchers import _DataLoaderIterDataFetcher
from pytorch_lightning.loops.utilities import _no_grad_context
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT

from careamics.prediction import stitch_prediction


class TiledPredictionLoop(L.loops._PredictionLoop):
    """Predict loop for tiles-based prediction."""

    # def _predict_step(self, batch, batch_idx, dataloader_idx, dataloader_iter):
    #     self.model.predict_step(batch, batch_idx)

    @_no_grad_context
    def run(self) -> Optional[_PREDICT_OUTPUT]:
        self.setup_data()
        if self.skip:
            return None
        self.reset()
        self.on_run_start()
        data_fetcher = self._data_fetcher
        assert data_fetcher is not None
        while True:
            try:
                if isinstance(data_fetcher, _DataLoaderIterDataFetcher):
                    dataloader_iter = next(data_fetcher)
                    # hook's batch_idx and dataloader_idx arguments correctness cannot be guaranteed in this setting
                    batch = data_fetcher._batch
                    batch_idx = data_fetcher._batch_idx
                    dataloader_idx = data_fetcher._dataloader_idx
                else:
                    dataloader_iter = None
                    batch, batch_idx, dataloader_idx = next(data_fetcher)
                self.batch_progress.is_last_batch = data_fetcher.done
                # run step hooks
                self._predict_step(batch, batch_idx, dataloader_idx, dataloader_iter)
            except StopIteration:
                # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
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

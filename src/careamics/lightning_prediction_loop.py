"""Lithning prediction loop allowing tiling."""

from typing import List, Optional

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.loops.fetchers import _DataLoaderIterDataFetcher
from pytorch_lightning.loops.utilities import _no_grad_context
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT

from careamics.config.tile_information import TileInformation
from careamics.dataset.tiling import stitch_prediction


class CAREamicsPredictionLoop(L.loops._PredictionLoop):
    """
    CAREamics prediction loop.

    This class extends the PyTorch Lightning `_PredictionLoop` class to include
    the stitching of the tiles into a single prediction result.
    """

    def _on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Call `on_predict_epoch_end` hook.

        Adapted from the parent method.

        Returns
        -------
        Optional[_PREDICT_OUTPUT]
            Prediction output.
        """
        trainer = self.trainer
        call._call_callback_hooks(trainer, "on_predict_epoch_end")
        call._call_lightning_module_hook(trainer, "on_predict_epoch_end")

        if self.return_predictions:
            ########################################################
            ################ CAREamics specific code ###############
            if len(self.predicted_array) == 1:
                # single array, already a numpy array
                return self.predicted_array[0]  # todo why not return the list here?
            else:
                return self.predicted_array
            ########################################################
        return None

    @_no_grad_context
    def run(self) -> Optional[_PREDICT_OUTPUT]:
        """Run the prediction loop.

        Adapted from the parent method in order to stitch the predictions.

        Returns
        -------
        Optional[_PREDICT_OUTPUT]
            Prediction output.
        """
        self.setup_data()
        if self.skip:
            return None
        self.reset()
        self.on_run_start()
        data_fetcher = self._data_fetcher
        assert data_fetcher is not None

        self.predicted_array = []
        self.tiles: List[np.ndarray] = []
        self.tile_information: List[TileInformation] = []

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

                ########################################################
                ################ CAREamics specific code ###############
                is_tiled = len(self.predictions[batch_idx]) == 2
                if is_tiled:
                    # a numpy array of shape BC(Z)YX
                    tile_batch = self.predictions[batch_idx][0]

                    # split the tiles into C(Z)YX (skip singleton S) and
                    # add them to the tiles list
                    self.tiles.extend(
                        [
                            tile[0]
                            for tile in np.split(
                                tile_batch.numpy(), tile_batch.shape[0], axis=0
                            )
                        ]
                    )

                    # list of TileInformation
                    tile_info = self.predictions[batch_idx][1]
                    self.tile_information.extend(tile_info)

                    # if last tile, stitch the tiles and add array to the prediction
                    last_tiles = [t.last_tile for t in self.tile_information]
                    if any(last_tiles):
                        sample_id_eq = [
                            t.sample_id == self.tile_information[0].sample_id
                            for t in self.tile_information
                        ]
                        sample_id = self.tile_information[0].sample_id
                        sample_tile_idx = [t.sample_id == sample_id for t in self.tile_information]
                        next_sample_tile_idx = [t.sample_id != sample_id for t in self.tile_information]
                        # # added for debug
                        # if not all(sample_id_eq):
                        #     raise ValueError("Not all tiles are from the same sample!")
                        predicted_batches = stitch_prediction(
                            [self.tiles[i] for i in sample_tile_idx], 
                            [self.tile_information[i] for i in sample_tile_idx]
                        )
                        self.predicted_array.append(predicted_batches)
                        self.tiles = [self.tiles[i] for i in next_sample_tile_idx]
                        self.tile_information = [self.tile_information[i] for i in next_sample_tile_idx]
                else:
                    # simply add the prediction to the list
                    self.predicted_array.append(self.predictions[batch_idx].numpy())
                ########################################################
            except StopIteration:
                break
            finally:
                self._restarting = False
        return self.on_run_end()

    # TODO predictions aren't stacked, list returned

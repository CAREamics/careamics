from typing import Optional

import pytorch_lightning as L
from pytorch_lightning.loops.fetchers import _DataLoaderIterDataFetcher
from pytorch_lightning.loops.utilities import _no_grad_context
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT

from careamics.prediction import stitch_prediction


class CAREamicsPredictionLoop(L.loops._PredictionLoop):
    """
    CAREamics prediction loop.

    This class extends the PyTorch Lightning `_PredictionLoop` class to include
    the stitching of the tiles into a single prediction result.
    """

    def _on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        """
        Calls `on_predict_epoch_end` hook.

        Adapted from the parent method.

        Returns
        -------
            the results for all dataloaders
        """
        trainer = self.trainer
        call._call_callback_hooks(trainer, "on_predict_epoch_end")
        call._call_lightning_module_hook(trainer, "on_predict_epoch_end")

        if self.return_predictions:
            ########################################################
            ################ CAREamics specific code ###############
            if len(self.predicted_array) == 1:
                # TODO does this make sense to here? (force numpy array)
                return self.predicted_array[0].numpy()
            else:
                # TODO revisit logic
                return [element.numpy() for element in self.predicted_array]
            ########################################################
        return None

    @_no_grad_context
    def run(self) -> Optional[_PREDICT_OUTPUT]:
        """
        Runs the prediction loop.

        Adapted from the parent method in order to stitch the predictions.

        Returns
        -------
        Optional[_PREDICT_OUTPUT]
            Prediction output
        """
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

                ########################################################
                ################ CAREamics specific code ###############
                is_tiled = len(self.predictions[batch_idx]) == 2
                if is_tiled:
                    # extract the last tile flag and the coordinates (crop and stitch)
                    last_tile, *stitch_data = self.predictions[batch_idx][1]

                    # append the tile and the coordinates to the lists
                    self.tiles.append(self.predictions[batch_idx][0])
                    self.stitching_data.append(stitch_data)

                    # if last tile, stitch the tiles and add array to the prediction
                    if any(last_tile):
                        predicted_batches = stitch_prediction(
                            self.tiles, self.stitching_data
                        )
                        self.predicted_array.append(predicted_batches)
                        self.tiles.clear()
                        self.stitching_data.clear()
                else:
                    # simply add the prediction to the list
                    self.predicted_array.append(self.predictions[batch_idx])
                ########################################################
            except StopIteration:
                break
            finally:
                self._restarting = False
        return self.on_run_end()

    # TODO predictions aren't stacked, list returned

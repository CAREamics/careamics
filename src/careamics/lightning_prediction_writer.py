from pathlib import Path
from typing import Any, Optional, Sequence

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from careamics.dataset.iterable_dataset import IterablePredictionDataset

from .lightning_module import CAREamicsModule
from .lightning_prediction_loop import CAREamicsPredictionLoop


class CAREamicsPredictionWriter(BasePredictionWriter):

    def __init__(
        self,
        write_interval="batch",
    ):
        super().__init__(write_interval)

    @staticmethod
    def _make_save_dir(pl_module: CAREamicsModule):
        predict_dir = pl_module.save_prediction_args.predict_dir
        predict_dir = Path(predict_dir)
        if not predict_dir.is_dir():
            predict_dir.mkdir()

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: CAREamicsModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        print("in write_on_batch_end.")
        if pl_module.save_prediction_args.save_predictions:
            print("saving to disk.")
            self._make_save_dir(pl_module)
            dl: DataLoader = trainer.predict_dataloaders
            ds: IterablePredictionDataset = dl.dataset
            files = ds.data_files
            loop: CAREamicsPredictionLoop = trainer.predict_loop
            
            # TODO: 
            # Get prediction array from loop
            # add prediction array clearing and also return file name if saving to disk
            
        else:
            print("Not saving to disk.")

import sys
from typing import Dict, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm


class ProgressBarCallback(TQDMProgressBar):
    """Progress bar for training and validation steps."""

    def init_train_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = tqdm(
            desc="Training",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.train_progress_bar is not None
        bar = tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout,
        )
        return bar

    def get_metrics(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        """Override this to customize the metrics displayed in the progress bar."""
        pbar_metrics = trainer.progress_bar_metrics
        return {**pbar_metrics}

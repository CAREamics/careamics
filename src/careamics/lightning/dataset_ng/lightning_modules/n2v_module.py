from typing import Any, Union

from careamics.config import (
    N2VAlgorithm,
)
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.losses import n2v_loss
from careamics.transforms import N2VManipulateTorch
from careamics.utils.logging import get_logger

from .unet_module import UnetModule

logger = get_logger(__name__)


class N2VModule(UnetModule):
    """CAREamics PyTorch Lightning module for N2V algorithm."""

    def __init__(self, algorithm_config: Union[N2VAlgorithm, dict]) -> None:
        super().__init__(algorithm_config)

        assert isinstance(
            algorithm_config, N2VAlgorithm
        ), "algorithm_config must be a N2VAlgorithm"

        self.n2v_manipulate = N2VManipulateTorch(
            n2v_manipulate_config=algorithm_config.n2v_config
        )
        self.loss_func = n2v_loss

    def _load_best_checkpoint(self) -> None:
        logger.warning(
            "Loading best checkpoint for N2V model. Note that for N2V, "
            "the checkpoint with the best validation metrics may not necessarily "
            "have the best denoising performance."
        )
        super()._load_best_checkpoint()

    def training_step(
        self,
        batch: Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]],
        batch_idx: Any,
    ) -> Any:
        """Training step for N2V model."""
        x = batch[0]
        x_masked, x_original, mask = self.n2v_manipulate(x.data)
        prediction = self.model(x_masked)
        loss = self.loss_func(prediction, x_original, mask)

        self._log_training_stats(loss, batch_size=x.data.shape[0])

        return loss

    def validation_step(
        self,
        batch: Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]],
        batch_idx: Any,
    ) -> None:
        """Validation step for N2V model."""
        x = batch[0]

        x_masked, x_original, mask = self.n2v_manipulate(x.data)
        prediction = self.model(x_masked)

        val_loss = self.loss_func(prediction, x_original, mask)
        self.metrics(prediction, x_original)
        self._log_validation_stats(val_loss, batch_size=x.data.shape[0])

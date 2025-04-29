from typing import Any, Callable, Union

from careamics.config.algorithms.care_algorithm_model import CAREAlgorithm
from careamics.config.algorithms.n2n_algorithm_model import N2NAlgorithm
from careamics.config.support import SupportedLoss
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.losses import mae_loss, mse_loss
from careamics.utils.logging import get_logger

from .unet_module import UnetModule

logger = get_logger(__name__)


class CAREModule(UnetModule):
    """CAREamics PyTorch Lightning module for CARE algorithm."""

    def __init__(self, algorithm_config: Union[CAREAlgorithm, dict]) -> None:
        super().__init__(algorithm_config)
        assert isinstance(
            algorithm_config, (CAREAlgorithm, N2NAlgorithm)
        ), "algorithm_config must be a CAREAlgorithm or a N2NAlgorithm"
        loss = algorithm_config.loss
        if loss == SupportedLoss.MAE:
            self.loss_func: Callable = mae_loss
        elif loss == SupportedLoss.MSE:
            self.loss_func = mse_loss
        else:
            raise ValueError(f"Unsupported loss for Care: {loss}")

    def training_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: Any,
    ) -> Any:
        """Training step for CARE module."""
        # TODO: add validation to determine if target is initialized
        x, target = batch[0], batch[1]

        prediction = self.model(x.data)
        loss = self.loss_func(prediction, target.data)

        self._log_training_stats(loss, batch_size=x.data.shape[0])

        return loss

    def validation_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: Any,
    ) -> None:
        """Validation step for CARE module."""
        x, target = batch[0], batch[1]

        prediction = self.model(x.data)
        val_loss = self.loss_func(prediction, target.data)
        self.metrics(prediction, target.data)
        self._log_validation_stats(val_loss, batch_size=x.data.shape[0])

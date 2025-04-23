from typing import Any, Union

from careamics.config.algorithms.care_algorithm_model import CAREAlgorithm
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.utils.logging import get_logger
from careamics.config.support import SupportedLoss
from careamics.losses import mae_loss, mse_loss

from .unet_module import UnetModule

logger = get_logger(__name__)


class CAREModule(UnetModule):
    def __init__(self, algorithm_config: Union[CAREAlgorithm, dict]) -> None:
        super().__init__(algorithm_config)
        assert isinstance(
            algorithm_config, CAREAlgorithm
        ), "algorithm_config must be a CAREAlgorithm"
        loss = algorithm_config.loss
        if loss == SupportedLoss.MAE:
            self.loss_func = mae_loss
        elif loss == SupportedLoss.MSE:
            self.loss_func = mse_loss
        else:
            raise ValueError(f"Unsupported loss for Care: {loss}")

    def training_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: Any,
    ) -> Any:
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
        x, target = batch[0], batch[1]

        prediction = self.model(x.data)
        val_loss = self.loss_func(prediction, target.data)
        self.metrics(prediction, target.data)
        self._log_validation_stats(val_loss, batch_size=x.data.shape[0])

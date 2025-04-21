from typing import Any, Union

from careamics.config.algorithms.care_algorithm_model import CAREAlgorithm
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.utils.logging import get_logger
from .base_module import BaseModule

logger = get_logger(__name__)


class CAREModule(BaseModule):
    def __init__(self, algorithm_config: Union[CAREAlgorithm, dict]) -> None:
        super().__init__(algorithm_config)
        assert isinstance(
            algorithm_config, CAREAlgorithm
        ), "algorithm_config must be a CAREAlgorithm"

    def training_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: Any,
    ) -> Any:
        # TODO: add validation to determine if target is initialized
        x, target = batch[0], batch[1]

        prediction = self.model(x.data)
        loss = self.loss_func(prediction, target.data)

        self._log_training_stats(loss)

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
        self._log_validation_stats(val_loss)

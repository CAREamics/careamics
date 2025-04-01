"""CAREamics Lightning module."""

from typing import Any, Optional, Union

import pytorch_lightning as L
from torch import Tensor, nn

from careamics.config import (
    N2VAlgorithm,
    UNetBasedAlgorithm,
    algorithm_factory,
)
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.losses import loss_factory
from careamics.models.model_factory import model_factory
from careamics.transforms import (
    Denormalize,
    N2VManipulateTorch,
)
from careamics.utils.torch_utils import get_optimizer, get_scheduler


class N2VModule(L.LightningModule):

    def __init__(self, algorithm_config: Union[UNetBasedAlgorithm, dict]) -> None:
        """Lightning module for CAREamics.

        This class encapsulates the a PyTorch model along with the training, validation,
        and testing logic. It is configured using an `AlgorithmModel` Pydantic class.

        Parameters
        ----------
        algorithm_config : AlgorithmModel or dict
            Algorithm configuration.
        """
        super().__init__()

        if isinstance(algorithm_config, dict):
            algorithm_config = algorithm_factory(algorithm_config)

        # create preprocessing, model and loss function
        if isinstance(algorithm_config, N2VAlgorithm):
            self.use_n2v = True
            self.n2v_preprocess: Optional[N2VManipulateTorch] = N2VManipulateTorch(
                n2v_manipulate_config=algorithm_config.n2v_config
            )
        else:
            self.use_n2v = False
            self.n2v_preprocess = None

        self.algorithm = algorithm_config.algorithm
        self.model: nn.Module = model_factory(algorithm_config.model)
        self.loss_func = loss_factory(algorithm_config.loss)

        # save optimizer and lr_scheduler names and parameters
        self.optimizer_name = algorithm_config.optimizer.name
        self.optimizer_params = algorithm_config.optimizer.parameters
        self.lr_scheduler_name = algorithm_config.lr_scheduler.name
        self.lr_scheduler_params = algorithm_config.lr_scheduler.parameters

    def forward(self, x: Any) -> Any:
        """Forward pass.

        Parameters
        ----------
        x : Any
            Input tensor.

        Returns
        -------
        Any
            Output tensor.
        """
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: Any) -> Any:
        """Training step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch.
        batch_idx : Any
            Batch index.

        Returns
        -------
        Any
            Loss value.
        """
        if len(batch.data) > 1:
            input_tensor, target = batch.data
        else:
            input_tensor = batch.data[0]
            target = None

        if self.use_n2v and self.n2v_preprocess is not None:
            masked_input, original_input, mask = self.n2v_preprocess(input_tensor)
            model_input = masked_input
            loss_args = (original_input, mask)
        else:
            model_input = input_tensor
            loss_args = (target.data,)

        prediction = self.model(model_input)
        loss = self.loss_func(prediction, *loss_args)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: Tensor, batch_idx: Any) -> None:
        """Validation step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch.
        batch_idx : Any
            Batch index.
        """
        if len(batch.data) > 1:
            input_tensor, target = batch.data
        else:
            input_tensor = batch.data[0]
            target = None

        if self.use_n2v and self.n2v_preprocess is not None:
            masked_input, original_input, mask = self.n2v_preprocess(input_tensor)
            model_input = masked_input
            loss_args = (original_input, mask)
        else:
            model_input = input_tensor
            loss_args = (target.data,)

        prediction = self.model(model_input)
        val_loss = self.loss_func(prediction, *loss_args)

        # log validation loss
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch: Tensor, batch_idx: Any) -> Any:
        if len(batch.data) > 1:
            input_tensor, target = batch.data
        else:
            input_tensor = batch.data[0]
            target = None

        output = self.model(input_tensor)

        denorm = Denormalize(
            image_means=self._trainer.datamodule.predict_dataset.input_stats.means,
            image_stds=self._trainer.datamodule.predict_dataset.input_stats.stds,
        )
        denormalized_output = denorm(patch=output.cpu().numpy())

        # if len(aux) > 0:  # aux can be tiling information
        #     return denormalized_output, *aux
        # else:
        #     return denormalized_output

        output_batch = ImageRegionData(
            data=denormalized_output,
            source=batch.source,
            data_shape=batch.data_shape,
            dtype=batch.dtype,
            axes=batch.axes,
            region_spec=batch.region_spec,
        )
        return output_batch

    def configure_optimizers(self) -> Any:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Any
            Optimizer and learning rate scheduler.
        """
        # instantiate optimizer
        optimizer_func = get_optimizer(self.optimizer_name)
        optimizer = optimizer_func(self.model.parameters(), **self.optimizer_params)

        # and scheduler
        scheduler_func = get_scheduler(self.lr_scheduler_name)
        scheduler = scheduler_func(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # otherwise triggers MisconfigurationException
        }


class UnetModule(L.LightningModule):
    pass

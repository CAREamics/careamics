"""CoLogger module for CAREamics."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lightning.fabric.utilities.logger import _flatten_dict
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import override

from careamics.config.configuration import Configuration
from careamics.config.utils.configuration_io import save_configuration

try:
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams

except ImportError:
    from tensorboardX import SummaryWriter
    from tensorboardX.summary import hparams


class CoLogger(Logger):
    """Combined Logger.

    This logger combines multiple loggers (CSV, TensorBoard, WandB)
    into a single interface.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    work_dir : Path
        Directory where logs will be saved.
    config : Configuration
        Configuration object containing experiment settings.
    use_tensorboard : bool, optional
        Whether to use TensorBoard for logging, by default False.
    use_wandb : bool, optional
        Whether to use WandB for logging, by default False.
    log_version : int, optional
        Version number for the logs, by default 0.
    finalize_after_fit : bool, optional
        Finalize and close loggers after `trainer.fit` finished, by default True.
    """

    def __init__(
        self,
        experiment_name: str,
        work_dir: Path,
        config: Configuration,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        log_version: int = 0,
        finalize_after_fit: bool = True,
    ) -> None:
        """Initialize the CoLogger.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        work_dir : Path
            Directory where logs will be saved.
        config : Configuration
            Configuration object containing experiment settings.
        use_tensorboard : bool, optional
            Whether to use TensorBoard for logging, by default False.
        use_wandb : bool, optional
            Whether to use WandB for logging, by default False.
        log_version : int, optional
            Version number for the logs, by default 0.
        finalize_after_fit : bool, optional
            Finalize and close loggers after `trainer.fit` finished, by default True.
        """
        super().__init__()

        self._name = experiment_name
        self._version = log_version
        self._log_dir = work_dir / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_log_dir = self._log_dir / "csv_logs"
        self._tb_log_dir = self._log_dir / "tb_logs"
        self._wandb_log_dir = self._log_dir / "wandb_logs"
        # flag to check if log_hyperparams has been called for the first time
        self._log_hp_first_call = True
        self.finalize_after_fit = finalize_after_fit
        self.loggers = []

        # save config as a yaml file
        save_configuration(config, self._log_dir)

        self.csv = CSVLogger(
            name=self._name,
            save_dir=self._csv_log_dir,
            version=self._version,
        )
        # add it to the list of loggers
        self.loggers.append(self.csv)

        self.tb: TensorBoardLogger | None = None
        if use_tensorboard:
            self.tb = TensorBoardLogger(
                name=self._name, save_dir=self._tb_log_dir, version=self._version
            )
            # add it to the list of loggers
            self.loggers.append(self.tb)

        self.wandb: WandbLogger | None = None
        if use_wandb:
            self.wandb = WandbLogger(
                name=self._name,
                save_dir=self._wandb_log_dir,
                config=config.model_dump(),
                version=str(self._version),
            )
            # add it to the list of loggers
            self.loggers.append(self.wandb)

    @rank_zero_only
    @override
    def log_hyperparams(
        self,
        params: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Log hyperparameters to all loggers.

        Parameters
        ----------
        params : dict[str, Any]
            A dictionary of hyperparameters to log.
        step : int | None, optional
            The current step or epoch number, by default None.
        """
        # we call this method again to add normalization stats
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                # tensorboard has issue with calling add_hparams multiple times
                if self._log_hp_first_call:
                    logger.log_hyperparams(params, metrics={}, step=step)
                else:
                    # update hparams dict
                    logger.hparams.update(params)
                    # update the hparams in tensorboard
                    self._update_tb_hparams(logger, step=step)
                # save the hparams.yaml file
                self._save_tb_hparams(logger)
            else:
                # csv and wandb
                logger.log_hyperparams(params)

        self._log_hp_first_call = False

    @rank_zero_only
    @override
    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Log metrics to all loggers.

        Parameters
        ----------
        metrics : dict[str, Any]
            A dictionary of metrics to log.
        step : int | None, optional
            The current step or epoch number, by default None.
        """
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step)

    @rank_zero_only
    def log_images(
        self,
        key: str,
        images: Tensor | NDArray,
        step: int | None = None,
        captions: list[str] | None = None,
        normalize: bool = True,
        axes: Literal["NCHW", "NHWC", "CHW", "HWC", "HW", "WH"] = "NCHW",
    ) -> None:
        """Log images to Tensorboard and/or WANDB.

        Parameters
        ----------
        key : str
            The key or tag for the images.
        images : Tensor | NDArray
            The images to log, either as a PyTorch tensor or a NumPy array.
        step : int | None, optional
            The current step or epoch number, by default None.
        captions : list[str] | None, optional
            A list of captions for the images, by default None.
        normalize : bool, optional
            Whether to normalize the images to [0, 255] range, by default True.
        axes : Literal["NCHW", "NHWC", "CHW", "HWC", "HW", "WH"], optional
            The axes of images to log, by default "NCHW".
        """
        if images.ndim != len(axes):
            raise ValueError(
                f"The images dimensions ({images.ndim}) "
                f"must represented in axes {axes}."
            )

        if self.tb is not None:
            self.tb.experiment.add_images(
                tag=key, img_tensor=images, global_step=step, dataformats=axes
            )

        if self.wandb is not None:
            # image stack to image list
            image_list = []
            if normalize:
                images = self._normalize_images(images)
            for i in range(len(images)):
                image_list.append(images[i])
            # handle extra args
            kwargs = {}
            if captions is not None:
                assert len(captions) == len(
                    image_list
                ), "You need a caption for each image!"
                kwargs["caption"] = captions

            self.wandb.log_image(key=key, images=image_list, step=step, **kwargs)

    @property
    @override
    def name(self) -> str:
        """Get the name of the logger.

        Returns
        -------
        str
            The name of the logger.
        """
        return self._name

    @property
    @override
    def version(self) -> int:
        """Get the version of the logger.

        Returns
        -------
        int
            The version of the logger.
        """
        return self._version

    @property
    @override
    def root_dir(self) -> Path:
        """Parent directory for all logs.

        Returns
        -------
        Path
            The path to the root directory for all logs.
        """
        return self._log_dir

    @property
    @override
    def log_dir(self) -> dict[str, str]:
        """List of directories for all loggers.

        Returns
        -------
        dict[str, str]
            A dictionary with logger names as keys and
            their corresponding log directories as values.
        """
        dirs = {"csv": self.csv.log_dir}
        if self.tb is not None:
            dirs["tensorboard"] = str(self._tb_log_dir.resolve())
        if self.wandb is not None:
            dirs["wandb"] = str(self._wandb_log_dir.resolve())

        return dirs

    @property
    @override
    def save_dir(self) -> Path:
        """The current directory where logs are saved.

        Returns
        -------
        Path
            The path to current directory where logs are saved.

        """
        return self._log_dir

    @override
    @rank_zero_only
    def save(self) -> None:
        """Save the state of all loggers."""
        super().save()
        self.csv.save()

    @rank_zero_only
    @override
    def finalize(self, status: str) -> None:
        """Finalize all loggers.

        Parameters
        ----------
        status : str
            The status of the training process (e.g., "success", "failure").
        """
        super().finalize(status)
        if not self.finalize_after_fit:
            # do not finalize
            return

        if self.tb is not None:
            self.tb.finalize(status)
        if self.wandb is not None:
            import wandb

            self.wandb.finalize(status)
            wandb.finish()

    @rank_zero_only
    def finish(self, status: str = "success") -> None:
        """Finalize and close all logger when `finalize_after_fit` is False.

        Parameters
        ----------
        status : str
            The status of the training process (e.g., "success", "failure").
        """
        self.finalize_after_fit = True
        self.finalize(status)

    def _save_tb_hparams(self, tb_logger: TensorBoardLogger):
        """Handle re-saving the hparams.yaml file for tensorboard logger.

        `tb_logger.save()` won't overwrite the yaml file.

        Parameters
        ----------
        tb_logger : TensorBoardLogger
            The TensorBoard logger instance.
        """
        # get tb log dir
        log_dir = Path(tb_logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # prepare the file path
        hparams_file = log_dir.joinpath("hparams.yaml")
        # save the yaml file
        save_hparams_to_yaml(hparams_file, tb_logger.hparams)

    def _update_tb_hparams(self, tb_logger: TensorBoardLogger, step: int | None = None):
        """Update the hparams in tensorboard logger.

        Parameters
        ----------
        tb_logger : TensorBoardLogger
            The TensorBoard logger instance.
        step : int | None, optional
            The current step or epoch number, by default None.
        """
        exp, ssi, sei = hparams(_flatten_dict(tb_logger.hparams), {})  # type: ignore
        with SummaryWriter(log_dir=tb_logger.log_dir) as w_hp:
            if w_hp.file_writer is not None:
                w_hp.file_writer.add_summary(exp)
                w_hp.file_writer.add_summary(ssi)
                w_hp.file_writer.add_summary(sei)
            else:
                raise ValueError("SummaryWriter `file_writer` is None!")

    def _normalize_images(self, images: NDArray | Tensor) -> Tensor:
        """Normalize images to [0, 255] range and convert to uint8.

        Parameters
        ----------
        images : NDArray | Tensor
            The images to normalize, either as a PyTorch tensor or a NumPy array.

        Returns
        -------
        NDArray
            The normalized images as a NumPy array of type uint8.
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        return (images * 255).to(torch.uint8)

"""
Engine module.

This module contains the main CAREamics class, the Engine. The Engine allows training
a model and using it for prediction.
"""
from logging import FileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from bioimageio.spec.model.raw_nodes import Model as BioimageModel
from torch.utils.data import DataLoader, TensorDataset

from .bioimage import (
    build_zip_model,
    get_default_model_specs,
)
from .config import Configuration, load_configuration
from .dataset.prepare_dataset import (
    get_prediction_dataset,
    get_train_dataset,
    get_validation_dataset,
)
from .losses import create_loss_function, create_noise_model
from .models import create_model
from .prediction import (
    stitch_prediction,
    tta_backward,
    tta_forward,
)
from .utils import (
    MetricTracker,
    check_array_validity,
    denormalize,
    get_device,
    normalize,
)
from .utils.logging import ProgressBar, get_logger


class Engine:
    """
    Class allowing training of a model and subsequent prediction.

    There are three ways to instantiate an Engine:
    1. With a CAREamics model (.pth), by passing a path.
    2. With a configuration object.
    3. With a configuration file, by passing a path.

    In each case, the parameter name must be provided explicitly. For example:
    >>> engine = Engine(config_path="path/to/config.yaml")

    Note that only one of these options can be used at a time, in the order listed
    above.

    Parameters
    ----------
    config : Optional[Configuration], optional
        Configuration object, by default None.
    config_path : Optional[Union[str, Path]], optional
        Path to configuration file, by default None.
    model_path : Optional[Union[str, Path]], optional
        Path to model file, by default None.
    seed : int, optional
        Seed for reproducibility, by default 42.

    Attributes
    ----------
    cfg : Configuration
        Configuration.
    device : torch.device
        Device (CPU or GPU).
    model : torch.nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        Optimizer.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler.
    loss_func : Callable
        Loss function.
    logger : logging.Logger
        Logger.
    use_wandb : bool
        Whether to use wandb.
    """

    def __init__(
        self,
        *,
        config: Optional[Configuration] = None,
        config_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Constructor.

        To disable the seed, set it to None.

        Parameters
        ----------
        config : Optional[Configuration], optional
            Configuration object, by default None.
        config_path : Optional[Union[str, Path]], optional
            Path to configuration file, by default None.
        model_path : Optional[Union[str, Path]], optional
            Path to model file, by default None.
        seed : int, optional
            Seed for reproducibility, by default 42.

        Raises
        ------
        ValueError
            If all three parameters are None.
        FileNotFoundError
            If the model or configuration path is provided but does not exist.
        TypeError
            If the configuration is not a Configuration object.
        UsageError
            If wandb is not correctly installed.
        ModuleNotFoundError
            If wandb is not installed.
        ValueError
            If the configuration failed to configure.
        """
        if model_path is not None:
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model path {model_path} is incorrect or"
                    f" does not exist. Current working directory is: {Path.cwd()!s}"
                )

            # Ensure that config is None
            self.cfg = None

        elif config is not None:
            # Check that config is a Configuration object
            if not isinstance(config, Configuration):
                raise TypeError(
                    f"config must be a Configuration object, got {type(config)}"
                )
            self.cfg = config
        elif config_path is not None:
            self.cfg = load_configuration(config_path)
        else:
            raise ValueError(
                "No configuration or path provided. One of configuration "
                "object, configuration path or model path must be provided."
            )

        # get device, CPU or GPU
        self.device = get_device()

        # Create model, optimizer, lr scheduler and gradient scaler and load everything
        # to the specified device
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.scaler,
            self.cfg,
        ) = create_model(config=self.cfg, model_path=model_path, device=self.device)

        # create loss function
        if self.cfg is not None:
            self.loss_func = create_loss_function(self.cfg)

            self.noise_model_blank = create_noise_model(self.cfg)

            # Set logging
            log_path = self.cfg.working_directory / "log.txt"
            self.logger = get_logger(__name__, log_path=log_path)

            # wandb
            self.use_wandb = self.cfg.training.use_wandb

            if self.use_wandb:
                try:
                    from wandb.errors import UsageError

                    from careamics.utils.wandb import WandBLogging

                    try:
                        self.wandb = WandBLogging(
                            experiment_name=self.cfg.experiment_name,
                            log_path=self.cfg.working_directory,
                            config=self.cfg,
                            model_to_watch=self.model,
                        )
                    except UsageError as e:
                        self.logger.warning(
                            f"Wandb usage error, using default logger. Check whether "
                            f"wandb correctly configured:\n"
                            f"{e}"
                        )
                        self.use_wandb = False

                except ModuleNotFoundError:
                    self.logger.warning(
                        "Wandb not installed, using default logger. Try pip install "
                        "wandb"
                    )
                    self.use_wandb = False
        else:
            raise ValueError("Configuration is not defined.")

    def train(
        self,
        train_path: str,
        val_path: str,
        train_target_path: Optional[str] = None,
        val_target_path: Optional[str] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Train the network.

        The training and validation data given by the paths must be compatible with the
        axes and data format provided in the configuration.

        Parameters
        ----------
        train_path : Union[str, Path]
            Path to the training data.
        val_path : Union[str, Path]
            Path to the validation data.

        Returns
        -------
        Tuple[List[Any], List[Any]]
            Tuple of training and validation statistics.

        Raises
        ------
        ValueError
            Raise a ValueError if the configuration is missing.
        """
        if self.cfg is None:
            raise ValueError("Configuration is not defined, cannot train.")

        if self.noise_model_blank is not None:
            self.noise_model = self.noise_model_blank(
                **self.cfg.algorithm.noise_model.parameters
            )

        train_loader = self._get_train_dataloader(train_path, train_target_path)
        eval_loader = self._get_val_dataloader(val_path, val_target_path)

        self.logger.info(f"Starting training for {self.cfg.training.num_epochs} epochs")

        val_losses = []

        try:
            train_stats = []
            eval_stats = []

            # loop over the dataset multiple times
            for epoch in range(self.cfg.training.num_epochs):
                if hasattr(train_loader.dataset, "__len__"):
                    epoch_size = train_loader.__len__()
                else:
                    try:
                        epoch_size = epoch_size
                    except NameError:
                        epoch_size = None

                progress_bar = ProgressBar(
                    max_value=epoch_size,
                    epoch=epoch,
                    num_epochs=self.cfg.training.num_epochs,
                    mode="train",
                )
                # Perform training step
                train_outputs, epoch_size = self._train_single_epoch(
                    train_loader,
                    progress_bar,
                    self.cfg.training.amp.use,
                )

                # Perform validation step
                eval_outputs = self._evaluate(eval_loader)
                val_losses.append(eval_outputs["loss"])
                learning_rate = self.optimizer.param_groups[0]["lr"]

                progress_bar.add(
                    1,
                    values=[
                        ("train_loss", train_outputs["loss"]),
                        ("val loss", eval_outputs["loss"]),
                        ("lr", learning_rate),
                    ],
                )
                # Add update scheduler rule based on type
                self.lr_scheduler.step(eval_outputs["loss"])

                if self.use_wandb:
                    metrics = {
                        "train": train_outputs,
                        "eval": eval_outputs,
                        "lr": learning_rate,
                    }
                    self.wandb.log_metrics(metrics)

                train_stats.append(train_outputs)
                eval_stats.append(eval_outputs)

                checkpoint_path = self._save_checkpoint(epoch, val_losses, "state_dict")
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Set mean and std from train dataset
            self.cfg.data.set_mean_and_std(
                train_loader.dataset.mean, train_loader.dataset.std
            )
            self.logger.info(
                f"Set mean/std to {train_loader.dataset.mean}, {train_loader.dataset.std}"
            )
        except KeyboardInterrupt:
            self.logger.info("Training interrupted")

        return train_stats, eval_stats

    def _train_single_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        progress_bar: Type[ProgressBar],
        amp: bool,
    ) -> Tuple[Dict[str, float], int]:
        """
        Train for a single epoch.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Training dataloader.
        progress_bar : ProgressBar
            Progress bar.
        amp : bool
            Whether to use automatic mixed precision.

        Returns
        -------
        Tuple[Dict[str, float], int]
            Tuple of training metrics and epoch size.

        Raises
        ------
        ValueError
            If the configuration is missing.
        """
        if self.cfg is not None:
            avg_loss = MetricTracker()
            self.model.train()
            epoch_size = 0

            for i, (batch, *auxillary) in enumerate(loader):
                self.optimizer.zero_grad(set_to_none=True)

                # Get prediction and loss
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = self.model(batch.to(self.device))
                loss = self.loss_func(
                    outputs, *[a.to(self.device) for a in auxillary], self.device
                )
                self.scaler.scale(loss).backward()
                avg_loss.update(loss.detach(), batch.shape[0])

                progress_bar.update(
                    current_step=i,
                    batch_size=self.cfg.training.batch_size,
                )

                self.optimizer.step()
                epoch_size += 1

            return {"loss": avg_loss.avg.to(torch.float16).cpu().numpy()}, epoch_size
        else:
            raise ValueError("Configuration is not defined, cannot train.")

    def _evaluate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Perform validation step.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            Validation dataloader.

        Returns
        -------
        Dict[str, float]
            Loss value on the validation set.
        """
        self.model.eval()
        avg_loss = MetricTracker()

        with torch.no_grad():
            for patch, *auxillary in val_loader:
                outputs = self.model(patch.to(self.device))
                loss = self.loss_func(
                    outputs, *[a.to(self.device) for a in auxillary], self.device
                )
                avg_loss.update(loss.detach(), patch.shape[0])
        return {"loss": avg_loss.avg.to(torch.float16).cpu().numpy()}

    def predict(
        self,
        input: Union[np.ndarray, str, Path],
        *,
        tile_shape: Optional[List[int]] = None,
        overlaps: Optional[List[int]] = None,
        axes: Optional[str] = None,
        tta: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predict using the current model on an input array or a path to data.

        The Engine must have previously been trained and mean/std be specified in
        its configuration.

        To use tiling, both `tile_shape` and `overlaps` must be specified, have same
        length, be divisible by 2 and greater than 0. Finally, the overlaps must be
        smaller than the tiles.

        Parameters
        ----------
        input : Union[np.ndarra, str, Path]
            Input data, either an array or a path to the data.
        tile_shape : Optional[List[int]], optional
            2D or 3D shape of the tiles to be predicted, by default None.
        overlaps : Optional[List[int]], optional
            2D or 3D overlaps between tiles, by default None.
        axes : Optional[str], optional
            Axes of the input array if different from the one in the configuration, by
            default None.
        tta : bool, optional
            Whether to use test time augmentation, by default True.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            Predicted image array of the same shape as the input, or list of arrays
            if the arrays have inconsistent shapes.

        Raises
        ------
        ValueError
            If the configuration is missing.
        ValueError
            If the mean or std are not specified in the configuration (untrained model).
        """
        if self.cfg is None:
            raise ValueError("Configuration is not defined, cannot predict.")

        # Check that the mean and std are there (= has been trained)
        if not self.cfg.data.mean or not self.cfg.data.std:
            raise ValueError(
                "Mean or std are not specified in the configuration, prediction cannot "
                "be performed."
            )

        # set model to eval mode
        self.model.to(self.device)
        self.model.eval()

        progress_bar = ProgressBar(num_epochs=1, mode="predict")

        # Get dataloader
        pred_loader, tiled = self._get_predict_dataloader(
            input=input, tile_shape=tile_shape, overlaps=overlaps, axes=axes
        )

        # Start prediction
        self.logger.info("Starting prediction")
        if tiled:
            self.logger.info("Starting tiled prediction")
            prediction = self._predict_tiled(pred_loader, progress_bar, tta)
        else:
            self.logger.info("Starting prediction on whole sample")
            prediction = self._predict_full(pred_loader, progress_bar, tta)

        return prediction

    def _predict_tiled(
        self, pred_loader: DataLoader, progress_bar: Type[ProgressBar], tta: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predict using tiling.

        Parameters
        ----------
        pred_loader : DataLoader
            Prediction dataloader.
        progress_bar : ProgressBar
            Progress bar.
        tta : bool, optional
            Whether to use test time augmentation, by default True.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            Predicted image, or list of predictions if the images have different sizes.

        Warns
        -----
        UserWarning
            If the samples have different shapes, the prediction then returns a list.
        """
        prediction = []
        tiles = []
        stitching_data = []

        with torch.no_grad():
            for i, (tile, *auxillary) in enumerate(pred_loader):
                # Unpack auxillary data into last tile indicator and data, required to
                # stitch tiles together
                if auxillary:
                    last_tile, *data = auxillary

                if tta:
                    augmented_tiles = tta_forward(tile)
                    predicted_augments = []
                    for augmented_tile in augmented_tiles:
                        augmented_pred = self.model(augmented_tile.to(self.device))
                        predicted_augments.append(
                            augmented_pred.squeeze().cpu().numpy()
                        )
                    tiles.append(tta_backward(predicted_augments).squeeze())
                else:
                    tiles.append(
                        self.model(tile.to(self.device)).squeeze().cpu().numpy()
                    )

                stitching_data.append(data)

                if last_tile:
                    # Stitch tiles together if sample is finished
                    predicted_sample = stitch_prediction(tiles, stitching_data)
                    predicted_sample = denormalize(
                        predicted_sample,
                        float(self.cfg.data.mean),  # type: ignore
                        float(self.cfg.data.std),  # type: ignore
                    )
                    prediction.append(predicted_sample)
                    tiles.clear()
                    stitching_data.clear()

                progress_bar.update(i, 1)

        self.logger.info(f"Predicted {len(prediction)} samples, {i} tiles in total")
        try:
            return np.stack(prediction)
        except ValueError:
            self.logger.warning("Samples have different shapes, returning list.")
            return prediction

    def _predict_full(
        self, pred_loader: DataLoader, progress_bar: type[ProgressBar], tta: bool = True
    ) -> np.ndarray:
        """
        Predict whole image without tiling.

        Parameters
        ----------
        pred_loader : DataLoader
            Prediction dataloader.
        progress_bar : ProgressBar
            Progress bar.
        tta : bool, optional
            Whether to use test time augmentation, by default True.

        Returns
        -------
        np.ndarray
            Predicted image.
        """
        prediction = []
        with torch.no_grad():
            for i, sample in enumerate(pred_loader):
                if tta:
                    augmented_preds = tta_forward(sample[0])
                    predicted_augments = []
                    for augmented_pred in augmented_preds:
                        augmented_pred = self.model(augmented_pred.to(self.device))
                        predicted_augments.append(
                            augmented_pred.squeeze().cpu().numpy()
                        )
                    prediction.append(tta_backward(predicted_augments).squeeze())
                else:
                    prediction.append(
                        self.model(sample[0].to(self.device)).squeeze().cpu().numpy()
                    )
                progress_bar.update(i, 1)
        output = denormalize(
            np.stack(prediction), float(self.cfg.data.mean), float(self.cfg.data.std)  # type: ignore
        )
        return output

    def _get_train_dataloader(
        self, train_path: str, train_target_path: str
    ) -> DataLoader:
        """
        Return a training dataloader.

        Parameters
        ----------
        train_path : str
            Path to the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        Raises
        ------
        ValueError
            If the training configuration is None.
        """
        if self.cfg is None:
            raise ValueError("Configuration is not defined.")

        dataset = get_train_dataset(self.cfg, train_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            pin_memory=False,
            # prefetch_factor=8,
        )
        return dataloader

    def _get_val_dataloader(self, val_path: str, val_target_path) -> DataLoader:
        """
        Return a validation dataloader.

        Parameters
        ----------
        val_path : str
            Path to the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        Raises
        ------
        ValueError
            If the configuration is None.
        """
        if self.cfg is None:
            raise ValueError("Configuration is not defined.")

        dataset = get_validation_dataset(self.cfg, val_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
        )
        return dataloader

    def _get_predict_dataloader(
        self,
        input: Union[np.ndarray, str, Path],
        *,
        tile_shape: Optional[List[int]] = None,
        overlaps: Optional[List[int]] = None,
        axes: Optional[str] = None,
    ) -> Tuple[DataLoader, bool]:
        """
        Return a prediction dataloader.

        Parameters
        ----------
        input : Union[np.ndarray, str, Path]
            Input array or path to data.
        tile_shape : Optional[List[int]], optional
            2D or 3D shape of the tiles, by default None.
        overlaps : Optional[List[int]], optional
            2D or 3D overlaps between tiles, by default None.
        axes : Optional[str], optional
            Axes of the input array if different from the one in the configuration.

        Returns
        -------
        Tuple[DataLoader, bool]
            Tuple of prediction data loader, and whether the data is tiled.

        Raises
        ------
        ValueError
            If the configuration is None.
        ValueError
            If the mean or std are not specified in the configuration.
        ValueError
            If the input is None.
        """
        if self.cfg is None:
            raise ValueError("Configuration is not defined.")

        if self.cfg.data.mean is None or self.cfg.data.std is None:
            raise ValueError(
                "Mean or std are not specified in the configuration, prediction cannot "
                "be performed. Was the model trained?"
            )

        if input is None:
            raise ValueError("Ccannot predict on None input.")

        # Create dataset
        if isinstance(input, np.ndarray):  # np.ndarray
            # Check that the axes fit the input
            img_axes = self.cfg.data.axes if axes is None else axes
            # TODO are self.cfg.data.axes and axes compatible (same spatial dim)?
            check_array_validity(input, img_axes)

            # Check if tiling requested
            tiled = tile_shape is not None and overlaps is not None

            # Validate tiles and overlaps
            if tiled:
                raise NotImplementedError(
                    "Tiling with in memory array is currently not implemented."
                )

                # check_tiling_validity(tile_shape, overlaps)

            # Normalize input and cast to float32
            normalized_input = normalize(
                img=input, mean=self.cfg.data.mean, std=self.cfg.data.std
            )
            normalized_input = normalized_input.astype(np.float32)

            # Create dataset
            dataset = TensorDataset(torch.from_numpy(normalized_input))

        elif isinstance(input, str) or isinstance(input, Path):  # path
            # Create dataset
            dataset = get_prediction_dataset(
                self.cfg,
                pred_path=input,
                tile_shape=tile_shape,
                overlaps=overlaps,
                axes=axes,
            )

            tiled = (
                hasattr(dataset, "patch_extraction_method")
                and dataset.patch_extraction_method is not None
            )
        return (
            DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=True,
            ),
            tiled,
        )

    def _save_checkpoint(
        self, epoch: int, losses: List[float], save_method: str
    ) -> Path:
        """
        Save checkpoint.

        Currently only supports saving using `save_method="state_dict"`.

        Parameters
        ----------
        epoch : int
            Last epoch.
        losses : List[float]
            List of losses.
        save_method : str
            Method to save the model. Currently only supports `state_dict`.

        Returns
        -------
        Path
            Path to the saved checkpoint.

        Raises
        ------
        ValueError
            If the configuration is None.
        NotImplementedError
            If the requested save method is not supported.
        """
        if self.cfg is None:
            raise ValueError("Configuration is not defined.")

        if epoch == 0 or losses[-1] == min(losses):
            name = f"{self.cfg.experiment_name}_best.pth"
        else:
            name = f"{self.cfg.experiment_name}_latest.pth"
        workdir = self.cfg.working_directory
        workdir.mkdir(parents=True, exist_ok=True)

        if save_method == "state_dict":
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "grad_scaler_state_dict": self.scaler.state_dict(),
                "loss": losses[-1],
                "config": self.cfg.model_dump(),
            }
            torch.save(checkpoint, workdir / name)
        else:
            raise NotImplementedError("Invalid save method.")

        return self.cfg.working_directory.absolute() / name

    def __del__(self) -> None:
        """Exit logger."""
        if hasattr(self, "logger"):
            for handler in self.logger.handlers:
                if isinstance(handler, FileHandler):
                    self.logger.removeHandler(handler)
                    handler.close()

    def _generate_rdf(self, model_specs: Optional[dict] = None) -> dict:
        """
        Generate rdf data for bioimage.io format export.

        Parameters
        ----------
        model_specs : Optional[dict], optional
            Custom specs if different than the default ones, by default None.

        Returns
        -------
        dict
            RDF specs.

        Raises
        ------
        ValueError
            If the mean or std are not specified in the configuration.
        ValueError
            If the configuration is not defined.
        """
        if self.cfg is not None:
            if self.cfg.data.mean is None or self.cfg.data.std is None:
                raise ValueError(
                    "Mean or std are not specified in the configuration, export to "
                    "bioimage.io format is not possible."
                )

            # set in/out axes from config
            axes = self.cfg.data.axes.lower().replace("s", "")
            if "c" not in axes:
                axes = "c" + axes
            if "b" not in axes:
                axes = "b" + axes

            # get in/out samples' files
            test_inputs, test_outputs = self._get_sample_io_files(axes)

            specs = get_default_model_specs(
                "Noise2Void",
                self.cfg.data.mean,
                self.cfg.data.std,
                self.cfg.algorithm.is_3D,
            )
            if model_specs is not None:
                specs.update(model_specs)

            specs.update(
                {
                    "architecture": "careamics.models.unet",
                    "test_inputs": test_inputs,
                    "test_outputs": test_outputs,
                    "input_axes": [axes],
                    "output_axes": [axes],
                }
            )
            return specs
        else:
            raise ValueError("Configuration is not defined.")

    def save_as_bioimage(
        self, output_zip: Union[Path, str], model_specs: Optional[dict] = None
    ) -> BioimageModel:
        """
        Export the current model to BioImage.io model zoo format.

        Parameters
        ----------
        output_zip : Union[Path, str]
            Where to save the model zip file.
        model_specs : Optional[dict]
            A dictionary with keys being the bioimage-core build_model parameters. If
            None then it will be populated by the model default specs.

        Returns
        -------
        BioimageModel
            Bioimage.io model object.

        Raises
        ------
        ValueError
            If the configuration is not defined.
        """
        if self.cfg is not None:
            # Generate specs
            specs = self._generate_rdf(model_specs)

            # Build model
            raw_model = build_zip_model(
                path=output_zip,
                config=self.cfg,
                model_specs=specs,
            )

            return raw_model
        else:
            raise ValueError("Configuration is not defined.")

    def _get_sample_io_files(self, axes: str) -> Tuple[List[str], List[str]]:
        """
        Create numpy format for use as inputs and outputs in the bioimage.io archive.

        Parameters
        ----------
        axes : str
            Input and output axes.

        Returns
        -------
        Tuple[List[str], List[str]]
            Tuple of input and output file paths.

        Raises
        ------
        ValueError
            If the configuration is not defined.
        """
        # input:
        if self.cfg is not None:
            sample_input = np.random.randn(*self.cfg.training.patch_size)
            # if there are more input axes (like channel, ...),
            # then expand the sample dimensions.
            len_diff = len(axes) - len(self.cfg.training.patch_size)
            if len_diff > 0:
                sample_input = np.expand_dims(
                    sample_input, axis=tuple(i for i in range(len_diff))
                )
            sample_output = np.random.randn(*sample_input.shape)

            # save numpy files
            workdir = self.cfg.working_directory
            in_file = workdir.joinpath("test_inputs.npy")
            np.save(in_file, sample_input)
            out_file = workdir.joinpath("test_outputs.npy")
            np.save(out_file, sample_output)

            return [str(in_file.absolute())], [str(out_file.absolute())]
        else:
            raise ValueError("Configuration is not defined.")

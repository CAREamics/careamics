"""A class to train, predict and export models in CAREamics."""

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from careamics.config import Configuration, FCNAlgorithmConfig, load_configuration
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedData,
    SupportedLogger,
)
from careamics.dataset.dataset_utils import reshape_array
from careamics.lightning import (
    FCNModule,
    HyperParametersCallback,
    PredictDataModule,
    ProgressBarCallback,
    TrainDataModule,
    create_predict_datamodule,
)
from careamics.model_io import export_to_bmz, load_pretrained
from careamics.prediction_utils import convert_outputs
from careamics.utils import check_path_exists, get_logger

logger = get_logger(__name__)

LOGGER_TYPES = Optional[Union[TensorBoardLogger, WandbLogger]]


class CAREamist:
    """Main CAREamics class, allowing training and prediction using various algorithms.

    Parameters
    ----------
    source : pathlib.Path or str or CAREamics Configuration
        Path to a configuration file or a trained model.
    work_dir : str, optional
        Path to working directory in which to save checkpoints and logs,
        by default None.
    callbacks : list of Callback, optional
        List of callbacks to use during training and prediction, by default None.

    Attributes
    ----------
    model : CAREamicsModule
        CAREamics model.
    cfg : Configuration
        CAREamics configuration.
    trainer : Trainer
        PyTorch Lightning trainer.
    experiment_logger : TensorBoardLogger or WandbLogger
        Experiment logger, "wandb" or "tensorboard".
    work_dir : pathlib.Path
        Working directory.
    train_datamodule : TrainDataModule
        Training datamodule.
    pred_datamodule : PredictDataModule
        Prediction datamodule.
    """

    @overload
    def __init__(  # numpydoc ignore=GL08
        self,
        source: Union[Path, str],
        work_dir: Optional[Union[Path, str]] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None: ...

    @overload
    def __init__(  # numpydoc ignore=GL08
        self,
        source: Configuration,
        work_dir: Optional[Union[Path, str]] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None: ...

    def __init__(
        self,
        source: Union[Path, str, Configuration],
        work_dir: Optional[Union[Path, str]] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        """
        Initialize CAREamist with a configuration object or a path.

        A configuration object can be created using directly by calling `Configuration`,
        using the configuration factory or loading a configuration from a yaml file.

        Path can contain either a yaml file with parameters, or a saved checkpoint.

        If no working directory is provided, the current working directory is used.

        Parameters
        ----------
        source : pathlib.Path or str or CAREamics Configuration
            Path to a configuration file or a trained model.
        work_dir : str or pathlib.Path, optional
            Path to working directory in which to save checkpoints and logs,
            by default None.
        callbacks : list of Callback, optional
            List of callbacks to use during training and prediction, by default None.

        Raises
        ------
        NotImplementedError
            If the model is loaded from BioImage Model Zoo.
        ValueError
            If no hyper parameters are found in the checkpoint.
        ValueError
            If no data module hyper parameters are found in the checkpoint.
        """
        # select current working directory if work_dir is None
        if work_dir is None:
            self.work_dir = Path.cwd()
            logger.warning(
                f"No working directory provided. Using current working directory: "
                f"{self.work_dir}."
            )
        else:
            self.work_dir = Path(work_dir)

        # configuration object
        if isinstance(source, Configuration):
            self.cfg = source

            # instantiate model
            if isinstance(self.cfg.algorithm_config, FCNAlgorithmConfig):
                self.model = FCNModule(
                    algorithm_config=self.cfg.algorithm_config,
                )
            else:
                raise NotImplementedError("Architecture not supported.")

        # path to configuration file or model
        else:
            source = check_path_exists(source)

            # configuration file
            if source.is_file() and (
                source.suffix == ".yaml" or source.suffix == ".yml"
            ):
                # load configuration
                self.cfg = load_configuration(source)

                # instantiate model
                if isinstance(self.cfg.algorithm_config, FCNAlgorithmConfig):
                    self.model = FCNModule(
                        algorithm_config=self.cfg.algorithm_config,
                    )  # type: ignore
                else:
                    raise NotImplementedError("Architecture not supported.")

            # attempt loading a pre-trained model
            else:
                self.model, self.cfg = load_pretrained(source)

        # define the checkpoint saving callback
        self._define_callbacks(callbacks)

        # instantiate logger
        if self.cfg.training_config.has_logger():
            if self.cfg.training_config.logger == SupportedLogger.WANDB:
                self.experiment_logger: LOGGER_TYPES = WandbLogger(
                    name=self.cfg.experiment_name,
                    save_dir=self.work_dir / Path("logs"),
                )
            elif self.cfg.training_config.logger == SupportedLogger.TENSORBOARD:
                self.experiment_logger = TensorBoardLogger(
                    save_dir=self.work_dir / Path("logs"),
                )
        else:
            self.experiment_logger = None

        # instantiate trainer
        self.trainer = Trainer(
            max_epochs=self.cfg.training_config.num_epochs,
            precision=self.cfg.training_config.precision,
            max_steps=self.cfg.training_config.max_steps,
            check_val_every_n_epoch=self.cfg.training_config.check_val_every_n_epoch,
            enable_progress_bar=self.cfg.training_config.enable_progress_bar,
            accumulate_grad_batches=self.cfg.training_config.accumulate_grad_batches,
            gradient_clip_val=self.cfg.training_config.gradient_clip_val,
            gradient_clip_algorithm=self.cfg.training_config.gradient_clip_algorithm,
            callbacks=self.callbacks,
            default_root_dir=self.work_dir,
            logger=self.experiment_logger,
        )

        # place holder for the datamodules
        self.train_datamodule: Optional[TrainDataModule] = None
        self.pred_datamodule: Optional[PredictDataModule] = None

    def _define_callbacks(self, callbacks: Optional[list[Callback]] = None) -> None:
        """Define the callbacks for the training loop.

        Parameters
        ----------
        callbacks : list of Callback, optional
            List of callbacks to use during training and prediction, by default None.
        """
        self.callbacks = [] if callbacks is None else callbacks

        # check that user callbacks are not any of the CAREamics callbacks
        for c in self.callbacks:
            if isinstance(c, ModelCheckpoint) or isinstance(c, EarlyStopping):
                raise ValueError(
                    "ModelCheckpoint and EarlyStopping callbacks are already defined "
                    "in CAREamics and should only be modified through the "
                    "training configuration (see TrainingConfig)."
                )

            if isinstance(c, HyperParametersCallback) or isinstance(
                c, ProgressBarCallback
            ):
                raise ValueError(
                    "HyperParameter and ProgressBar callbacks are defined internally "
                    "and should not be passed as callbacks."
                )

        # checkpoint callback saves checkpoints during training
        self.callbacks.extend(
            [
                HyperParametersCallback(self.cfg),
                ModelCheckpoint(
                    dirpath=self.work_dir / Path("checkpoints"),
                    filename=self.cfg.experiment_name,
                    **self.cfg.training_config.checkpoint_callback.model_dump(),
                ),
                ProgressBarCallback(),
            ]
        )

        # early stopping callback
        if self.cfg.training_config.early_stopping_callback is not None:
            self.callbacks.append(
                EarlyStopping(self.cfg.training_config.early_stopping_callback)
            )

    def stop_training(self) -> None:
        """Stop the training loop."""
        # raise stop training flag
        self.trainer.should_stop = True
        self.trainer.limit_val_batches = 0  # skip  validation

    # TODO: is there are more elegant way than calling train again after _train_on_paths
    def train(
        self,
        *,
        datamodule: Optional[TrainDataModule] = None,
        train_source: Optional[Union[Path, str, NDArray]] = None,
        val_source: Optional[Union[Path, str, NDArray]] = None,
        train_target: Optional[Union[Path, str, NDArray]] = None,
        val_target: Optional[Union[Path, str, NDArray]] = None,
        use_in_memory: bool = True,
        val_percentage: float = 0.1,
        val_minimum_split: int = 1,
    ) -> None:
        """
        Train the model on the provided data.

        If a datamodule is provided, then training will be performed using it.
        Alternatively, the training data can be provided as arrays or paths.

        If `use_in_memory` is set to True, the source provided as Path or str will be
        loaded in memory if it fits. Otherwise, training will be performed by loading
        patches from the files one by one. Training on arrays is always performed
        in memory.

        If no validation source is provided, then the validation is extracted from
        the training data using `val_percentage` and `val_minimum_split`. In the case
        of data provided as Path or str, the percentage and minimum number are applied
        to the number of files. For arrays, it is the number of patches.

        Parameters
        ----------
        datamodule : TrainDataModule, optional
            Datamodule to train on, by default None.
        train_source : pathlib.Path or str or NDArray, optional
            Train source, if no datamodule is provided, by default None.
        val_source : pathlib.Path or str or NDArray, optional
            Validation source, if no datamodule is provided, by default None.
        train_target : pathlib.Path or str or NDArray, optional
            Train target source, if no datamodule is provided, by default None.
        val_target : pathlib.Path or str or NDArray, optional
            Validation target source, if no datamodule is provided, by default None.
        use_in_memory : bool, optional
            Use in memory dataset if possible, by default True.
        val_percentage : float, optional
            Percentage of validation extracted from training data, by default 0.1.
        val_minimum_split : int, optional
            Minimum number of validation (patch or file) extracted from training data,
            by default 1.

        Raises
        ------
        ValueError
            If both `datamodule` and `train_source` are provided.
        ValueError
            If sources are not of the same type (e.g. train is an array and val is
            a Path).
        ValueError
            If the training target is provided to N2V.
        ValueError
            If neither a datamodule nor a source is provided.
        """
        if datamodule is not None and train_source is not None:
            raise ValueError(
                "Only one of `datamodule` and `train_source` can be provided."
            )

        # check that inputs are the same type
        source_types = {
            type(s)
            for s in (train_source, val_source, train_target, val_target)
            if s is not None
        }
        if len(source_types) > 1:
            raise ValueError("All sources should be of the same type.")

        # train
        if datamodule is not None:
            self._train_on_datamodule(datamodule=datamodule)

        else:
            # raise error if target is provided to N2V
            if self.cfg.algorithm_config.algorithm == SupportedAlgorithm.N2V.value:
                if train_target is not None:
                    raise ValueError(
                        "Training target not compatible with N2V training."
                    )

            # dispatch the training
            if isinstance(train_source, np.ndarray):
                # mypy checks
                assert isinstance(val_source, np.ndarray) or val_source is None
                assert isinstance(train_target, np.ndarray) or train_target is None
                assert isinstance(val_target, np.ndarray) or val_target is None

                self._train_on_array(
                    train_source,
                    val_source,
                    train_target,
                    val_target,
                    val_percentage,
                    val_minimum_split,
                )

            elif isinstance(train_source, Path) or isinstance(train_source, str):
                # mypy checks
                assert (
                    isinstance(val_source, Path)
                    or isinstance(val_source, str)
                    or val_source is None
                )
                assert (
                    isinstance(train_target, Path)
                    or isinstance(train_target, str)
                    or train_target is None
                )
                assert (
                    isinstance(val_target, Path)
                    or isinstance(val_target, str)
                    or val_target is None
                )

                self._train_on_path(
                    train_source,
                    val_source,
                    train_target,
                    val_target,
                    use_in_memory,
                    val_percentage,
                    val_minimum_split,
                )

            else:
                raise ValueError(
                    f"Invalid input, expected a str, Path, array or TrainDataModule "
                    f"instance (got {type(train_source)})."
                )

    def _train_on_datamodule(self, datamodule: TrainDataModule) -> None:
        """
        Train the model on the provided datamodule.

        Parameters
        ----------
        datamodule : TrainDataModule
            Datamodule to train on.
        """
        # register datamodule
        self.train_datamodule = datamodule

        # set defaults (in case `stop_training` was called before)
        self.trainer.should_stop = False
        self.trainer.limit_val_batches = 1.0  # 100%

        # train
        self.trainer.fit(self.model, datamodule=datamodule)

    def _train_on_array(
        self,
        train_data: NDArray,
        val_data: Optional[NDArray] = None,
        train_target: Optional[NDArray] = None,
        val_target: Optional[NDArray] = None,
        val_percentage: float = 0.1,
        val_minimum_split: int = 5,
    ) -> None:
        """
        Train the model on the provided data arrays.

        Parameters
        ----------
        train_data : NDArray
            Training data.
        val_data : NDArray, optional
            Validation data, by default None.
        train_target : NDArray, optional
            Train target data, by default None.
        val_target : NDArray, optional
            Validation target data, by default None.
        val_percentage : float, optional
            Percentage of patches to use for validation, by default 0.1.
        val_minimum_split : int, optional
            Minimum number of patches to use for validation, by default 5.
        """
        # create datamodule
        datamodule = TrainDataModule(
            data_config=self.cfg.data_config,
            train_data=train_data,
            val_data=val_data,
            train_data_target=train_target,
            val_data_target=val_target,
            val_percentage=val_percentage,
            val_minimum_split=val_minimum_split,
        )

        # train
        self.train(datamodule=datamodule)

    def _train_on_path(
        self,
        path_to_train_data: Union[Path, str],
        path_to_val_data: Optional[Union[Path, str]] = None,
        path_to_train_target: Optional[Union[Path, str]] = None,
        path_to_val_target: Optional[Union[Path, str]] = None,
        use_in_memory: bool = True,
        val_percentage: float = 0.1,
        val_minimum_split: int = 1,
    ) -> None:
        """
        Train the model on the provided data paths.

        Parameters
        ----------
        path_to_train_data : pathlib.Path or str
            Path to the training data.
        path_to_val_data : pathlib.Path or str, optional
            Path to validation data, by default None.
        path_to_train_target : pathlib.Path or str, optional
            Path to train target data, by default None.
        path_to_val_target : pathlib.Path or str, optional
            Path to validation target data, by default None.
        use_in_memory : bool, optional
            Use in memory dataset if possible, by default True.
        val_percentage : float, optional
            Percentage of files to use for validation, by default 0.1.
        val_minimum_split : int, optional
            Minimum number of files to use for validation, by default 1.
        """
        # sanity check on data (path exists)
        path_to_train_data = check_path_exists(path_to_train_data)

        if path_to_val_data is not None:
            path_to_val_data = check_path_exists(path_to_val_data)

        if path_to_train_target is not None:
            path_to_train_target = check_path_exists(path_to_train_target)

        if path_to_val_target is not None:
            path_to_val_target = check_path_exists(path_to_val_target)

        # create datamodule
        datamodule = TrainDataModule(
            data_config=self.cfg.data_config,
            train_data=path_to_train_data,
            val_data=path_to_val_data,
            train_data_target=path_to_train_target,
            val_data_target=path_to_val_target,
            use_in_memory=use_in_memory,
            val_percentage=val_percentage,
            val_minimum_split=val_minimum_split,
        )

        # train
        self.train(datamodule=datamodule)

    @overload
    def predict(  # numpydoc ignore=GL08
        self, source: PredictDataModule
    ) -> Union[list[NDArray], NDArray]: ...

    @overload
    def predict(  # numpydoc ignore=GL08
        self,
        source: Union[Path, str],
        *,
        batch_size: int = 1,
        tile_size: Optional[tuple[int, ...]] = None,
        tile_overlap: tuple[int, ...] = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["tiff", "custom"]] = None,
        tta_transforms: bool = False,
        dataloader_params: Optional[dict] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
    ) -> Union[list[NDArray], NDArray]: ...

    @overload
    def predict(  # numpydoc ignore=GL08
        self,
        source: NDArray,
        *,
        batch_size: int = 1,
        tile_size: Optional[tuple[int, ...]] = None,
        tile_overlap: tuple[int, ...] = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["array"]] = None,
        tta_transforms: bool = False,
        dataloader_params: Optional[dict] = None,
    ) -> Union[list[NDArray], NDArray]: ...

    def predict(
        self,
        source: Union[PredictDataModule, Path, str, NDArray],
        *,
        batch_size: Optional[int] = None,
        tile_size: Optional[tuple[int, ...]] = None,
        tile_overlap: Optional[tuple[int, ...]] = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["array", "tiff", "custom"]] = None,
        tta_transforms: bool = False,
        dataloader_params: Optional[dict] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        **kwargs: Any,
    ) -> Union[list[NDArray], NDArray]:
        """
        Make predictions on the provided data.

        Input can be a CAREamicsPredData instance, a path to a data file, or a numpy
        array.

        If `data_type`, `axes` and `tile_size` are not provided, the training
        configuration parameters will be used, with the `patch_size` instead of
        `tile_size`.

        Test-time augmentation (TTA) can be switched off using the `tta_transforms`
        parameter. The TTA augmentation applies all possible flip and 90 degrees
        rotations to the prediction input and averages the predictions. TTA augmentation
        should not be used if you did not train with these augmentations.

        Note that if you are using a UNet model and tiling, the tile size must be
        divisible in every dimension by 2**d, where d is the depth of the model. This
        avoids artefacts arising from the broken shift invariance induced by the
        pooling layers of the UNet. If your image has less dimensions, as it may
        happen in the Z dimension, consider padding your image.

        Parameters
        ----------
        source : CAREamicsPredData, pathlib.Path, str or numpy.ndarray
            Data to predict on.
        batch_size : int, default=1
            Batch size for prediction.
        tile_size : tuple of int, optional
            Size of the tiles to use for prediction.
        tile_overlap : tuple of int, default=(48, 48)
            Overlap between tiles, can be None.
        axes : str, optional
            Axes of the input data, by default None.
        data_type : {"array", "tiff", "custom"}, optional
            Type of the input data.
        tta_transforms : bool, default=True
            Whether to apply test-time augmentation.
        dataloader_params : dict, optional
            Parameters to pass to the dataloader.
        read_source_func : Callable, optional
            Function to read the source data.
        extension_filter : str, default=""
            Filter for the file extension.
        **kwargs : Any
            Unused.

        Returns
        -------
        list of NDArray or NDArray
            Predictions made by the model.

        Raises
        ------
        ValueError
            If mean and std are not provided in the configuration.
        ValueError
            If tile size is not divisible by 2**depth for UNet models.
        ValueError
            If tile overlap is not specified.
        """
        if (
            self.cfg.data_config.image_means is None
            or self.cfg.data_config.image_stds is None
        ):
            raise ValueError("Mean and std must be provided in the configuration.")

        # tile size for UNets
        if tile_size is not None:
            model = self.cfg.algorithm_config.model

            if model.architecture == SupportedArchitecture.UNET.value:
                # tile size must be equal to k*2^n, where n is the number of pooling
                # layers (equal to the depth) and k is an integer
                depth = model.depth
                tile_increment = 2**depth

                for i, t in enumerate(tile_size):
                    if t % tile_increment != 0:
                        raise ValueError(
                            f"Tile size must be divisible by {tile_increment} along "
                            f"all axes (got {t} for axis {i}). If your image size is "
                            f"smaller along one axis (e.g. Z), consider padding the "
                            f"image."
                        )

            # tile overlaps must be specified
            if tile_overlap is None:
                raise ValueError("Tile overlap must be specified.")

        # create the prediction
        self.pred_datamodule = create_predict_datamodule(
            pred_data=source,
            data_type=data_type or self.cfg.data_config.data_type,
            axes=axes or self.cfg.data_config.axes,
            image_means=self.cfg.data_config.image_means,
            image_stds=self.cfg.data_config.image_stds,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size or self.cfg.data_config.batch_size,
            tta_transforms=tta_transforms,
            read_source_func=read_source_func,
            extension_filter=extension_filter,
            dataloader_params=dataloader_params,
        )

        # predict
        predictions = self.trainer.predict(
            model=self.model, datamodule=self.pred_datamodule
        )
        return convert_outputs(predictions, self.pred_datamodule.tiled)

    def export_to_bmz(
        self,
        path_to_archive: Union[Path, str],
        friendly_model_name: str,
        input_array: NDArray,
        authors: list[dict],
        general_description: str = "",
        channel_names: Optional[list[str]] = None,
        data_description: Optional[str] = None,
    ) -> None:
        """Export the model to the BioImage Model Zoo format.

        This method packages the current weights into a zip file that can be uploaded
        to the BioImage Model Zoo. The archive consists of the model weights, the model
        specifications and various files (inputs, outputs, README, env.yaml etc.).

        `path_to_archive` should point to a file with a ".zip" extension.

        `friendly_model_name` is the name used for the model in the BMZ specs
        and website, it should consist of letters, numbers, dashes, underscores and
        parentheses only.

        Input array must be of the same dimensions as the axes recorded in the
        configuration of the `CAREamist`.

        Parameters
        ----------
        path_to_archive : pathlib.Path or str
            Path in which to save the model, including file name, which should end with
            ".zip".
        friendly_model_name : str
            Name of the model as used in the BMZ specs, it should consist of letters,
            numbers, dashes, underscores and parentheses only.
        input_array : NDArray
            Input array used to validate the model and as example.
        authors : list of dict
            List of authors of the model.
        general_description : str
            General description of the model, used in the metadata of the BMZ archive.
        channel_names : list of str, optional
            Channel names, by default None.
        data_description : str, optional
            Description of the data, by default None.
        """
        # TODO: add in docs that it is expected that input_array dimensions match
        # those in data_config

        output_patch = self.predict(
            input_array,
            data_type=SupportedData.ARRAY.value,
            tta_transforms=False,
        )
        output = np.concatenate(output_patch, axis=0)
        input_array = reshape_array(input_array, self.cfg.data_config.axes)

        export_to_bmz(
            model=self.model,
            config=self.cfg,
            path_to_archive=path_to_archive,
            model_name=friendly_model_name,
            general_description=general_description,
            authors=authors,
            input_array=input_array,
            output_array=output,
            channel_names=channel_names,
            data_description=data_description,
        )

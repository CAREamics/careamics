"""Main class to train and predict with CAREamics models."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch import load

from .config import (
    AlgorithmModel,
    Configuration,
    DataModel,
    TrainingModel,
    create_inference_configuration,
    load_configuration,
)
from .config.inference_model import TRANSFORMS_UNION
from .config.support import SupportedAlgorithm
from .lightning_datamodule import CAREamicsClay, CAREamicsWood
from .lightning_module import CAREamicsKiln
from .lightning_prediction import CAREamicsPredictionLoop
from .utils import check_path_exists, get_logger

logger = get_logger(__name__)

# TODO napari callbacks
# TODO save as modelzoo, lightning and pytorch_dict
# TODO load checkpoints
# TODO how to do WandB
# TODO: how to do AMP? How to continue training? How to load model from checkpoint?
# TODO: how to save checkpoints?


class CAREamist(LightningModule):
    """
    A class to train and predict with CAREamics models.

    # TODO

    Parameters
    ----------
    source : Union[Path, str, Configuration]
        Path to a configuration file or a trained model.
    work_dir : Optional[str], optional
        Path to working directory in which to save checkpoints and logs,
        by default None.
    experiment_name : str, optional
        Experiment name used for checkpoints, by default "CAREamics".

    """

    @overload
    def __init__(  # numpydoc ignore=GL08
        self,
        source: Union[Path, str],
        work_dir: Optional[str] = None,
        experiment_name: str = "CAREamics",
    ) -> None:
        ...

    @overload
    def __init__(  # numpydoc ignore=GL08
        self,
        source: Configuration,
        work_dir: Optional[str] = None,
        experiment_name: str = "CAREamics",
    ) -> None:
        ...

    def __init__(
        self,
        source: Union[Path, str, Configuration],
        work_dir: Optional[Union[Path, str]] = None,
        experiment_name: str = "CAREamics",
    ) -> None:
        """
        Initialize CAREamist with a configuration object or a path.

        A configuration object can be created using directly by calling `Configuration`,
        using the configuration factory or loading a configuration from a yaml file.

        Path can contain either a yaml file with parameters, or a saved checkpoint.

        If no working directory is provided, the current working directory is used.

        If `source` is a checkpoint, then `experiment_name` is used to name the
        checkpoint, and is recorded in the configuration.

        Parameters
        ----------
        source : Union[Path, str, Configuration]
            Path to a configuration file or a trained model.
        work_dir : Optional[str], optional
            Path to working directory in which to save checkpoints and logs,
            by default None.
        experiment_name : str, optional
            Experiment name used for checkpoints, by default "CAREamics".

        Raises
        ------
        NotImplementedError
            If the model is loaded from BioImage Model Zoo.
        ValueError
            If no hyper parameters are found in the checkpoint.
        ValueError
            If no data module hyper parameters are found in the checkpoint.

        """
        super().__init__()

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
            self.model = CAREamicsKiln(self.cfg.algorithm)

        # path to configuration file or model
        else:
            source = check_path_exists(source)

            # configuration file
            if source.is_file() and (
                source.suffix == ".yaml" or source.suffix == ".yml"
            ):
                # load configuration
                self.cfg = load_configuration(source)

                # save configuration in the working directory
                # TODO should be in train
                self.save_hyperparameters(self.cfg.model_dump())

                # instantiate model
                self.model = CAREamicsKiln(self.cfg.algorithm)

            # bmz model
            elif source.suffix == ".zip":
                raise NotImplementedError(
                    "Loading a model from BioImage Model Zoo is not implemented yet."
                )

            # checkpoint
            elif source.suffix == ".ckpt":
                checkpoint = load(source)

                # attempt to load algorithm parameters
                try:
                    self.algo_params = checkpoint["hyper_parameters"]
                except KeyError as e:
                    raise ValueError(
                        "Invalid checkpoint file. No `hyper_parameters` found for the "
                        "algorithm."
                    ) from e

                # attempt to load data model parameters
                try:
                    self.data_params = checkpoint["datamodule_hyper_parameters"]
                except KeyError as e:
                    raise ValueError(
                        "Invalid checkpoint file. No `datamodule_hyper_parameters` "
                        "found for the data."
                    ) from e

                # create configuration
                algorithm = AlgorithmModel(**self.algo_params)
                data = DataModel(**self.data_params)
                training = TrainingModel()
                self.cfg = Configuration(
                    experiment_name=experiment_name,
                    algorithm=algorithm,
                    data=data,
                    training=training,
                )

                # load weights
                self.load_pretrained(checkpoint)

        # define the checkpoint saving callback
        self.callbacks = self._define_callbacks()

        # instantiate trainer
        self.trainer = Trainer(
            max_epochs=self.cfg.training.num_epochs,
            callbacks=self.callbacks,
            default_root_dir=self.work_dir,
        )

        # change the prediction loop, necessary for tiled prediction
        self.trainer.predict_loop = CAREamicsPredictionLoop(self.trainer)

    def _define_callbacks(self) -> List[Callback]:
        """
        Define the callbacks for the training loop.

        Returns
        -------
        List[Callback]
            List of callbacks to be used during training.
        """
        # checkpoint callback saves checkpoints during training
        self.callbacks = [
            ModelCheckpoint(
                dirpath=self.work_dir / Path("checkpoints"),
                filename=self.cfg.experiment_name,
                **self.cfg.training.checkpoint_callback.model_dump(),
            )
        ]

        # early stopping callback
        if self.cfg.training.early_stopping_callback is not None:
            self.callbacks.append(
                EarlyStopping(self.cfg.training.early_stopping_callback)
            )

        return self.callbacks

    def train(
        self,
        *,
        datamodule: Optional[CAREamicsWood] = None,
        train_source: Optional[Union[Path, str, np.ndarray]] = None,
        val_source: Optional[Union[Path, str, np.ndarray]] = None,
        train_target: Optional[Union[Path, str, np.ndarray]] = None,
        val_target: Optional[Union[Path, str, np.ndarray]] = None,
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
        datamodule : Optional[CAREamicsWood], optional
            Datamodule to train on, by default None.
        train_source : Optional[Union[Path, str, np.ndarray]], optional
            Train source, if no datamodule is provided, by default None.
        val_source : Optional[Union[Path, str, np.ndarray]], optional
            Validation source, if no datamodule is provided, by default None.
        train_target : Optional[Union[Path, str, np.ndarray]], optional
            Train target source, if no datamodule is provided, by default None.
        val_target : Optional[Union[Path, str, np.ndarray]], optional
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
        if datamodule is not None and train_source:
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
            if self.cfg.algorithm.algorithm == SupportedAlgorithm.N2V.value:
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
                    f"Invalid input, expected a str, Path, array or CAREamicsWood "
                    f"instance (got {type(train_source)})."
                )

    def _train_on_datamodule(self, datamodule: CAREamicsWood) -> None:
        """
        Train the model on the provided datamodule.

        Parameters
        ----------
        datamodule : CAREamicsWood
            Datamodule to train on.
        """
        self.trainer.fit(self.model, datamodule=datamodule)

    def _train_on_array(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        train_target: Optional[np.ndarray] = None,
        val_target: Optional[np.ndarray] = None,
        val_percentage: float = 0.1,
        val_minimum_split: int = 5,
    ) -> None:
        """
        Train the model on the provided data arrays.

        Parameters
        ----------
        train_data : np.ndarray
            Training data.
        val_data : Optional[np.ndarray], optional
            Validation data, by default None.
        train_target : Optional[np.ndarray], optional
            Train target data, by default None.
        val_target : Optional[np.ndarray], optional
            Validation target data, by default None.
        val_percentage : float, optional
            Percentage of patches to use for validation, by default 0.1.
        val_minimum_split : int, optional
            Minimum number of patches to use for validation, by default 5.
        """
        # create datamodule
        datamodule = CAREamicsWood(
            data_config=self.cfg.data,
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
        path_to_train_data : Union[Path, str]
            Path to the training data.
        path_to_val_data : Optional[Union[Path, str]], optional
            Path to validation data, by default None.
        path_to_train_target : Optional[Union[Path, str]], optional
            Path to train target data, by default None.
        path_to_val_target : Optional[Union[Path, str]], optional
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
        datamodule = CAREamicsWood(
            data_config=self.cfg.data,
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
        self, source: CAREamicsClay
    ) -> Union[list, np.ndarray]:
        ...

    @overload
    def predict(  # numpydoc ignore=GL08
        self,
        source: Union[Path, str],
        *,
        batch_size: int = 1,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: Tuple[int, ...] = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["tiff", "custom"]] = None,
        transforms: Optional[List[TRANSFORMS_UNION]] = None,
        tta_transforms: bool = True,
        dataloader_params: Optional[Dict] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
    ) -> Union[list, np.ndarray]:
        ...

    @overload
    def predict(  # numpydoc ignore=GL08
        self,
        source: np.ndarray,
        *,
        batch_size: int = 1,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: Tuple[int, ...] = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["array"]] = None,
        transforms: Optional[List[TRANSFORMS_UNION]] = None,
        tta_transforms: bool = True,
        dataloader_params: Optional[Dict] = None,
    ) -> Union[list, np.ndarray]:
        ...

    def predict(
        self,
        source: Union[CAREamicsClay, Path, str, np.ndarray],
        *,
        batch_size: int = 1,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: Tuple[int, ...] = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["array", "tiff", "custom"]] = None,
        transforms: Optional[List[TRANSFORMS_UNION]] = None,
        tta_transforms: bool = True,
        dataloader_params: Optional[Dict] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        **kwargs: Any,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """
        Make predictions on the provided data.

        Input can be a CAREamicsClay instance, a path to a data file, or a numpy array.

        If `data_type`, `axes` and `tile_size` are not provided, the training
        configuration parameters will be used, with the `patch_size` instead of
        `tile_size`.

        The default transforms are defined in the InferenceModel Pydantic model.

        TTA transform can be switched off using the `tta_transforms` parameter.

        Parameters
        ----------
        source : Union[CAREamicsClay, Path, str, np.ndarray]
            Data to predict on.
        batch_size : int, optional
            Batch size for prediction, by default 1.
        tile_size : Optional[Tuple[int, ...]], optional
            Size of the tiles to use for prediction, by default None.
        tile_overlap : Tuple[int, ...], optional
            Overlap between tiles, by default (48, 48).
        axes : Optional[str], optional
            Axes of the input data, by default None.
        data_type : Optional[Literal["array", "tiff", "custom"]], optional
            Type of the input data, by default None.
        transforms : Optional[List[TRANSFORMS_UNION]], optional
            List of transforms to apply to the data, by default None.
        tta_transforms : bool, optional
            Whether to apply test-time augmentation, by default True.
        dataloader_params : Optional[Dict], optional
            Parameters to pass to the dataloader, by default None.
        read_source_func : Optional[Callable], optional
            Function to read the source data, by default None.
        extension_filter : str, optional
            Filter for the file extension, by default "".
        **kwargs : Any
            Unused.

        Returns
        -------
        Union[List[np.ndarray], np.ndarray]
            Predictions made by the model.

        Raises
        ------
        ValueError
            If the input is not a CAREamicsClay instance, a path or a numpy array.
        """
        if isinstance(source, CAREamicsClay):
            return self.trainer.predict(datamodule=source)

        else:
            # create predict config, reuse training config if parameters missing
            prediction_config = create_inference_configuration(
                training_configuration=self.cfg,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                data_type=data_type,
                axes=axes,
                transforms=transforms,
                tta_transforms=tta_transforms,
                batch_size=batch_size,
            )

            # remove batch from dataloader parameters (priority given to config)
            if dataloader_params is None:
                dataloader_params = {}
            if "batch_size" in dataloader_params:
                del dataloader_params["batch_size"]

            if isinstance(source, Path) or isinstance(source, str):
                # Check the source
                source_path = check_path_exists(source)

                # create datamodule
                datamodule = CAREamicsClay(
                    prediction_config=prediction_config,
                    pred_data=source_path,
                    read_source_func=read_source_func,
                    extension_filter=extension_filter,
                    dataloader_params=dataloader_params,
                )

                return self.trainer.predict(datamodule=datamodule)

            elif isinstance(source, np.ndarray):
                # create datamodule
                datamodule = CAREamicsClay(
                    prediction_config=prediction_config,
                    pred_data=source,
                    dataloader_params=dataloader_params,
                )

                return self.trainer.predict(datamodule=datamodule)

            else:
                raise ValueError(
                    f"Invalid input. Expected a CAREamicsWood instance, paths or "
                    f"np.ndarray (got {type(source)})."
                )

    def export_checkpoint(
        self, path: Union[Path, str], type: Literal["bmz", "script"] = "bmz"
    ) -> None:
        """
        Export the model to a checkpoint or a BioImage Model Zoo model.

        Parameters
        ----------
        path : Union[Path, str]
            Path to save the model.
        type : Literal["bmz", "script"], optional
            Export format, by default "bmz".

        Raises
        ------
        NotImplementedError
            If the export format is not implemented yet.
        """
        path = Path(path)
        if type == "bmz":
            raise NotImplementedError(
                "Exporting a model to BioImage Model Zoo is not implemented yet."
            )
        elif type == "script":
            self.model.to_torchscript(path)
        else:
            raise ValueError(
                f"Invalid export format. Expected 'bmz' or 'script', got {type}."
            )

    def load_pretrained(self, path: Union[Path, str]) -> None:
        """
        Load a pretrained model from a checkpoint or a BioImage Model Zoo model.

        Expected formats are .ckpt, .zip, .pth or .pt files.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the pretrained model.

        Raises
        ------
        ValueError
            If the model format is not supported.
        """
        path = check_path_exists(path)

        if path.suffix == ".ckpt":
            self._load_from_checkpoint(path)
        elif path.suffix == ".zip":
            self._load_from_bmz(path)
        elif path.suffix == ".pth" or path.suffix == ".pt":
            self._load_from_state_dict(path)
        else:
            raise ValueError(
                f"Invalid model format. Expected .ckpt, .zip, .pth or .pt file, "
                f"got {path.suffix}."
            )

    def _load_from_checkpoint(self, path: Union[Path, str]) -> None:
        """
        Load a model from a checkpoint.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the checkpoint.
        """
        self.model.load_from_checkpoint(path)

    def _load_from_bmz(
        self,
        path: Union[Path, str],
    ) -> None:
        """
        Load a model from BioImage Model Zoo.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the BioImage Model Zoo model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented yet.
        """
        raise NotImplementedError(
            "Loading a model from BioImage Model Zoo is not implemented yet."
        )

    def _load_from_state_dict(
        self,
        path: Union[Path, str],
    ) -> None:
        """
        Load a model from a state dict.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the state dict.

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """
        raise NotImplementedError(
            "Loading a model from a state dict is not implemented yet."
        )

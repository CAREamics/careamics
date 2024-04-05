from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, overload

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
from .lightning_prediction import CAREamicsFiring
from .utils import check_path_exists, get_logger

logger = get_logger(__name__)

# TODO callbacks
# TODO save as modelzoo, lightning and pytorch_dict
# TODO load checkpoints
# TODO validation set from training set
# TODO train and predict on np.ndarray
# TODO how to do WandB
# TODO: how to do AMP? How to continue training? How to load model from checkpoint?
# TODO: how to save checkpoints?
# TODO configure training parameters (epochs, etc.), potentially needs to be possible here


class CAREamist(LightningModule):
    """A class to train and predict with CAREamics models.

    There are three ways to instantiate the CAREamist class:
        - with a path to a BioImage Model Zoo model (BMZ format)
        - with a Configuration object (see Configuration model)
        - with a path to a configuration file

    One of these parameters must be provided. If multiple parameters are passed,
    then the priority is set following the list above: model > configuration > path.

    Parameters
    ----------
    path_to_model : Optional[Union[Path, str]], optional
        Path to a BioImge Model Zoo model on disk, by default None
    configuration : Optional[Configuration], optional
        Configuration object, by default None
    path_to_config : Optional[Union[Path, str]], optional
        Path to a configuration yaml file, by default None

    Raises
    ------
    TypeError
        If configuration is not a Configuration object
    FileNotFoundError
        If the path to the configuration file does not exist
    ValueError
        If the path is not pointing to a file
    ValueError
        If no configuration or path is provided
    """

    @overload
    def __init__(
        self,
        source: Union[Path, str],
        work_dir: Optional[str] = None,
        experiment_name: str = "CAREamics",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        source: Configuration,
        work_dir: Optional[str] = None,
        experiment_name: str = "CAREamics",
    ) -> None:
        ...

    def __init__(
        self,
        source: Union[Path, str, Configuration],
        work_dir: Optional[str] = None,
        experiment_name: str = "CAREamics",
    ) -> None:
        """Initialize CAREamist with a configuration object or a path.

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
            by default None
        experiment_name : str, optional
            Experiment name used for checkpoints, by default "CAREamics"

        Raises
        ------
        NotImplementedError
            _description_ #TODO
        ValueError
            _description_
        ValueError
            _description_
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
            self.work_dir = work_dir

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
        self.trainer.predict_loop = CAREamicsFiring(self.trainer)

    def _define_callbacks(self) -> List[Callback]:
        """Define the callbacks for the training loop.

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

        # TODO
        """
        #
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
                self._train_on_array(
                    train_source,
                    val_source,
                    train_target,
                    val_target,
                    val_percentage,
                    val_minimum_split,
                )

            elif isinstance(train_source, Path) or isinstance(train_source, str):
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
        self.trainer.fit(self.model, datamodule=datamodule)

    def _train_on_array(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        train_target: Optional[np.ndarray] = None,
        val_target: Optional[np.ndarray] = None,
        val_percentage: float = 0.1,
        val_minimum_split: int = 1,
    ) -> None:
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
    def predict(self, source: CAREamicsClay) -> Union[list, np.ndarray]:
        ...

    @overload
    def predict(
        self,
        source: Union[Path, str],
        *,
        batch_size: int = 1,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: int = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["tiff", "custom"]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: str = "",
        transforms: Optional[List[TRANSFORMS_UNION]] = None,
        tta_transforms: bool = True,
        dataloader_params: Optional[Dict] = None,
    ) -> Union[list, np.ndarray]:
        if dataloader_params is None:
            dataloader_params = {}
        ...

    @overload
    def predict(
        self,
        source: np.ndarray,
        *,
        batch_size: int = 1,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: int = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["array"]] = None,
        transforms: Optional[List[TRANSFORMS_UNION]] = None,
        tta_transforms: bool = True,
        dataloader_params: Optional[Dict] = None,
    ) -> Union[list, np.ndarray]:
        if dataloader_params is None:
            dataloader_params = {}
        ...

    def predict(
        self,
        source: Union[CAREamicsClay, Path, str, np.ndarray],
        *,
        batch_size: int = 1,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: int = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["array", "tiff", "custom"]] = None,
        read_source_func: Optional[Callable] = None,
        extension_filter: Optional[str] = "",
        transforms: Optional[List[TRANSFORMS_UNION]] = None,
        tta_transforms: bool = True,
        dataloader_params: Optional[Dict] = None,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Make predictions on the provided data.

        Input can be a CAREamicsClay instance, a path to a data file, or a numpy array.

        # TODO

        Parameters
        ----------
        source : Union[CAREamicsClay, Path, str, np.ndarray]
            Data to predict on.

        Returns
        -------
        Union[List[np.ndarray], np.ndarray]
            Predictions made by the model.

        Raises
        ------
        ValueError
            If the input is not a CAREamicsClay instance, a path or a numpy array.
        """
        if dataloader_params is None:
            dataloader_params = {}
        if isinstance(source, CAREamicsClay):
            return self.trainer.predict(datamodule=datamodule)

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
        path = Path(path)
        if type == "bmz":
            raise NotImplementedError(
                "Exporting a model to BioImage Model Zoo is not implemented yet."
            )
        elif type == "script":
            self.model.to_torchscript(path)

    def load_pretrained(self, path: Union[Path, str]) -> None:
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

    def _load_from_checkpoint(self, path):
        self.model.load_from_checkpoint(path)

    def _load_from_bmz(
        self,
        path: Union[Path, str],
    ):
        raise NotImplementedError(
            "Loading a model from BioImage Model Zoo is not implemented yet."
        )

    def _load_from_state_dict(
        self,
        path: Union[Path, str],
    ):
        raise NotImplementedError(
            "Loading a model from a state dict is not implemented yet."
        )

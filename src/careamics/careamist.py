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
    InferenceModel,
    TrainingModel,
    create_inference_configuration,
    load_configuration,
)
from .config.support import SupportedAlgorithm
from .lightning_module import CAREamicsKiln
from .lightning_prediction import CAREamicsFiring
from .lightning_datamodule import CAREamicsClay, CAREamicsWood
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
            _description_
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

    def train(self, *args, **kwargs) -> None:
        if len(args) > 0:
            raise ValueError(
                "Only keyword arguments are allowed for the `train` method."
            )
        if any(isinstance(p, CAREamicsWood) for p in kwargs.values()):
            try:
                datamodule = kwargs["datamodule"]
            except KeyError:
                print("An instance of CAREamicsWood must be provided.")

            self.train_on_datamodule(datamodule=datamodule)

        elif all(isinstance(p, Path) for p in kwargs.values()):
            self.train_on_path(*args, **kwargs)

        elif all(isinstance(p, str) for p in kwargs.values()):
            self._train_on_str(*args, **kwargs)

        elif all(isinstance(p, np.ndarray) for p in kwargs.values()):
            self.train_on_array(*args, **kwargs)

        else:
            raise ValueError(
                "Invalid input. Expected a CAREamicsWood instance, paths or np.ndarray."
            )

    def train_on_datamodule(
        self,
        datamodule: CAREamicsWood,
    ) -> None:
        self.trainer.fit(self.model, datamodule=datamodule)

    def train_on_path(
        self,
        path_to_train_data: Union[Path, str],
        path_to_val_data: Optional[Union[Path, str]] = None,
        path_to_train_target: Optional[Union[Path, str]] = None,
        path_to_val_target: Optional[Union[Path, str]] = None,
        use_in_memory: bool = True,
    ) -> None:
        # sanity check on data (path exists)
        path_to_train_data = check_path_exists(path_to_train_data)

        if path_to_val_data is not None:
            path_to_val_data = check_path_exists(path_to_val_data)

        if path_to_train_target is not None:
            if self.cfg.algorithm.algorithm != SupportedAlgorithm.N2V.value:
                raise ValueError(
                    f"Training target is not needed for unsupervised algorithms "
                    f"({self.cfg.algorithm.algorithm})."
                )

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
        )

        # train
        self.train(datamodule=datamodule)

    def train_on_array(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        train_target: Optional[np.ndarray] = None,
        val_target: Optional[np.ndarray] = None,
    ) -> None:
        if train_target is not None:
            if self.cfg.algorithm.algorithm != SupportedAlgorithm.N2V.value:
                raise ValueError(
                    f"Training target is not needed for unsupervised algorithms "
                    f"({self.cfg.algorithm.algorithm})."
                )

        # create datamodule
        datamodule = CAREamicsWood(
            data_config=self.cfg.data,
            train_data=train_data,
            val_data=val_data,
            train_data_target=train_target,
            val_data_target=val_target,
        )

        # train
        self.train(datamodule=datamodule)

    @overload
    def predict(
        self, source: CAREamicsClay
    ) -> Union[list, np.ndarray]:
        ...

    @overload
    def predict(
        self, source: Union[Path, str]
    ) -> Union[list, np.ndarray]:
        ...

    @overload
    def predict(
        self, source: np.ndarray
    ) -> Union[list, np.ndarray]:
        ...

    def predict(
        self,
        source,
        *,
        batch_size: int = 1,
        tile_size: Optional[Tuple[int, ...]] = None,
        tile_overlap: int = (48, 48),
        axes: Optional[str] = None,
        data_type: Optional[Literal["array", "tiff", "custom"]] = None,
        read_source_func: Optional[Callable] = None,
        transforms: Optional[List] = None,
        tta_transforms: Optional[bool] = True,
        extension_filter: Optional[str] = "",
        dataloader_params: Optional[Dict] = None,
    ) -> Union[list, np.ndarray]:
        """Make predictions on the provided data.

        Input can be a CAREamicsClay instance, a path to a data file, or a numpy array.

        Parameters
        ----------
        source : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if isinstance(source, CAREamicsClay):
            return self.trainer.predict(datamodule=source)

        elif isinstance(source, Path) or isinstance(source, str):
            # Check the source
            source_path = check_path_exists(source)

            # create predict config, reuse training config if parameters are not provided
            prediction_config = create_inference_configuration(
                training_configuration=self.cfg,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                data_type=data_type,
                axes=axes,
                transforms=transforms,
                tta_transforms=tta_transforms
            )
            # create datamodule
            datamodule = CAREamicsClay(
                prediction_config=prediction_config,
                pred_data=source_path,
                read_source_func=read_source_func,
                extension_filter=extension_filter,
                dataloader_params=dataloader_params,
            )

            return self._predict_on_datamodule(datamodule)

        elif isinstance(source, np.ndarray):
            self._predict_on_array(data=source, prediction_config=prediction_config)

        else:
            raise ValueError(
                "Invalid input. Expected a CAREamicsWood instance, paths or np.ndarray."
            )

    def _predict_on_datamodule(
        self,
        datamodule: CAREamicsClay,
    ) -> None:
        preds = self.trainer.predict(datamodule=datamodule)
        return preds

    def _predict_on_path(
        self,
        path_to_data: Path,
        prediction_config: InferenceModel,
    ) -> Dict[str, np.ndarray]:
        # sanity check (path exists)
        path = check_path_exists(path_to_data)
        """
        create predict config, reuse the training config if parameters are not provided
        remove all prediction specific parameters from data_config
        """
        # create datamodule
        datamodule = CAREamicsClay(
            data_config=prediction_config,
            pred_data=path,
        )

        return self._predict_on_datamodule(datamodule)

    def _predict_on_str(
        self,
        str_to_data: str,
        prediction_config: InferenceModel,
    ) -> Dict[str, np.ndarray]:
        path_to_data = Path(str_to_data)

        return self._predict_on_path(path_to_data, prediction_config)

    def _predict_on_array(
        self,
        data: np.ndarray,
        tile_size: Tuple[int, ...],
        tile_overlap: Tuple[int, ...],
    ) -> Dict[str, np.ndarray]:
        # create datamodule
        datamodule = CAREamicsClay(
            prediction_config=self.cfg.data,
            pred_data=data,
        )

        return self.predict(datamodule)

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

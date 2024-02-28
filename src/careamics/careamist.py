from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from pytorch_lightning import Trainer

from .config import Configuration, load_configuration
from .lightning_module import CAREamicsKiln
from .lightning_prediction import CAREamicsFiring
from .ligthning_datamodule import CAREamicsClay, CAREamicsWood
from .utils import check_path_exists, method_dispatch


# TODO callbacks
# TODO save as modelzoo, lightning and pytorch_dict
# TODO load checkpoints
# TODO validation set from training set
# TODO train and predict on np.ndarray
# TODO how to do WandB
# TODO: how to do AMP? How to continue training? How to load model from checkpoint?
# TODO: how to save checkpoints?
# TODO configure training parameters (epochs, etc.), potentially needs to be possible here
class CAREamist:
    def __init__(
        self,
        *,
        path_to_model: Optional[Union[Path, str]] = None,
        configuration: Optional[Configuration] = None,
        path_to_config: Optional[Union[Path, str]] = None,
    ) -> None:
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
        if path_to_model is not None:
            raise NotImplementedError(
                "Loading a model from BioImage Model Zoo is not implemented yet."
            )
        elif configuration is not None:
            # Check that config is a Configuration object
            if not isinstance(configuration, Configuration):
                raise TypeError(
                    f"`config` must be a Configuration object, "
                    f"got {type(configuration)}"
                )

            self.cfg = configuration

        elif path_to_config is not None:
            path_to_config = Path(path_to_config)
            if not path_to_config.exists():
                raise FileNotFoundError(
                    f"Configuration path {path_to_config} does not exist."
                )
            elif not path_to_config.is_file():
                raise ValueError(f"Configuration path {path_to_config} is not a file.")

            # load configuration
            self.cfg = load_configuration(path_to_config)

        else:
            raise ValueError(
                "One of `path_to_model`, `configuration` or `path_to_config` "
                "must be provided to the CAREamist."
            )

        # instantiate model
        self.model = CAREamicsKiln(self.cfg.algorithm)

        # instantiate trainer
        self.trainer = Trainer(max_epochs=self.cfg.training.num_epochs)

        # change the prediction loop
        self.trainer.predict_loop = CAREamicsFiring(self.trainer)

    @method_dispatch
    def train(
        self,
        datamodule: CAREamicsWood,
    ) -> None:
        if not isinstance(datamodule, CAREamicsWood):
            raise TypeError(
                f"`datamodule` must be a CAREamicsWood instance, "
                f"got {type(datamodule)}."
            )

        self.trainer.fit(self.model, datamodule=datamodule)

    @train.register
    def _train_on_path(
        self,
        path_to_train_data: Path,  # cannot use Union annotation for the dispatch
        path_to_val_data: Optional[Path] = None,
        path_to_train_target: Optional[Path] = None,
        path_to_val_target: Optional[Path] = None,
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
        )

        # train
        self.train(datamodule=datamodule)

    @train.register
    def _train_on_str(
        self,
        path_to_train_data: str,
        path_to_val_data: Optional[str] = None,
        path_to_train_target: Optional[str] = None,
        path_to_val_target: Optional[str] = None,
    ) -> None:
        self._train_on_path(
            Path(path_to_train_data),
            Path(path_to_val_data) if path_to_val_data is not None else None,
            Path(path_to_train_target) if path_to_train_target is not None else None,
            Path(path_to_val_target) if path_to_val_target is not None else None,
        )

    @train.register
    def _train_on_array(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        train_target: Optional[np.ndarray] = None,
        val_target: Optional[np.ndarray] = None,
    ) -> None:
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

    @method_dispatch
    def predict(
        self,
        datamodule: CAREamicsClay,
    ) -> Dict[str, np.ndarray]:
        if not isinstance(datamodule, CAREamicsClay):
            raise TypeError(
                f"`datamodule` must be a CAREamicsClay instance, "
                f"got {type(datamodule)}."
            )

        return self.trainer.predict(self.model, datamodule=datamodule)

    @predict.register
    def _predict_on_path(
        self,
        path_to_data: Path,
    ) -> Dict[str, np.ndarray]:
        # sanity check (path exists)
        path = check_path_exists(path_to_data)

        # create datamodule
        datamodule = CAREamicsClay(
            data_config=self.cfg.data,
            pred_data=path,
        )

        return self.predict(datamodule)

    @predict.register
    def _predict_on_str(
        self,
        path_to_data: str,
    ) -> Dict[str, np.ndarray]:
        path_to_data = Path(path_to_data)

        return self._predict_on_path(path_to_data)

    @predict.register
    def _predict_on_array(
        self,
        data: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        # create datamodule
        datamodule = CAREamicsClay(
            data_config=self.cfg.data,
            pred_data=data,
        )

        return self.predict(datamodule)

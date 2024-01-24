from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader

from careamics.config import Configuration, load_configuration
from careamics.lightning import CAREamicsModel
from careamics.dataset.prepare_dataset import (
    get_train_dataset,
    get_validation_dataset,
    get_prediction_dataset,
)


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
        configuration: Optional[Configuration] = None,
        path_to_config: Optional[Union[Path, str]] = None,
    ) -> None:
        """A class to train and predict with CAREamics models.

        There are three ways to instantiate the CAREamist class:
            - with a Configuration object (see Configuration model)
            - with a path to a configuration file

        One of these parameters must be provided. If multiple parameters are passed,
        then the priority is set following the list above: model > configuration > path.

        Parameters
        ----------
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
        if configuration is not None:
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
                raise ValueError(
                    f"Configuration path {path_to_config} is not a file."
                )

            # load configuration            
            self.cfg = load_configuration(path_to_config)

        else:
            raise ValueError(
                "No configuration or path provided. One of configuration "
                "object or path must be provided."
            )

        # instantiate model
        self.model = CAREamicsModel(self.cfg.algorithm)

        # instantiate trainer
        self.trainer = Trainer(max_epochs=self.cfg.training.num_epochs)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def train_on_path(
        self,
        path_to_train_data: Union[Path, str],
        path_to_val_data: Union[Path, str],
    ) -> None:
        # sanity check on train data
        path_to_train_data = Path(path_to_train_data)
        if not path_to_train_data.exists():
            raise FileNotFoundError(
                f"Data path {path_to_train_data} is incorrect or"
                f" does not exist."
            )
        elif not path_to_train_data.is_dir():
            raise ValueError(
                f"Data path {path_to_train_data} is not a directory."
            )
        
        # sanity check on val data
        path_to_val_data = Path(path_to_val_data)
        if not path_to_val_data.exists():
            raise FileNotFoundError(
                f"Data path {path_to_val_data} is incorrect or"
                f" does not exist."
            )
        elif not path_to_val_data.is_dir():
            raise ValueError(
                f"Data path {path_to_val_data} is not a directory."
            )

        # create datasets and dataloaders
        train_dataset = get_train_dataset(self.cfg.data, path_to_train_data)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=0#self.cfg.training.num_workers,
        )

        val_dataset = get_validation_dataset(self.cfg.data, path_to_val_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=0,
        )

        # train
        self.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)


    def predict(
        self,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, np.ndarray]:

        return self.trainer.predict(self.model, test_dataloader)

    def predict_on_path(
        self,
        path_to_data: Union[Path, str],
        tile_shape: Optional[tuple] = None,
        overlaps: Optional[tuple] = None,
    ) -> Dict[str, np.ndarray]:
        path = Path(path_to_data)
        if not path.exists():
            raise FileNotFoundError(
                f"Data path {path_to_data} is incorrect or"
                f" does not exist."
            )
        elif not path.is_dir():
            raise ValueError(
                f"Data path {path_to_data} is not a directory."
            )

        # create dataset
        pred_dataset = get_prediction_dataset(
            self.cfg.data, 
            path_to_data,
            tile_shape=tile_shape,
            overlaps=overlaps,
            )
        
        # create dataloader
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
        )

        # TODO how to deal with stitching?

        # predict
        return self.predict(pred_dataloader)


    def save(
        self,
        format: str = "modelzoo",  # TODO Enum
    ):
        raise NotImplementedError(
            "Saving is not implemented yet."
        )

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from careamics.config import Configuration, load_configuration
from careamics.lightning import LUNet, CAREamicsModel
from careamics.dataset.prepare_dataset import (
    get_train_dataset,
    get_validation_dataset,
    get_prediction_dataset,
)


# TODO: throughout the code, we need to pass the submodels of the configuration
class CAREamist:
    def __init__(
        self,
        *,
        configuration: Optional[Configuration] = None,
        path_to_config: Optional[Union[Path, str]] = None,
        path_to_model: Optional[Union[Path, str]] = None,
    ) -> None:
        if path_to_model is not None:
            # if not Path(path_to_model).exists():
            #     raise FileNotFoundError(
            #         f"Model path {path_to_model} is incorrect or"
            #         f" does not exist. Current working directory is: {Path.cwd()!s}"
            #     )

            # # Ensure that config is None
            # self.cfg = None
            # TODO implement
            raise NotImplementedError("Loading checkpoint not yet implemented.")

        elif configuration is not None:
            # Check that config is a Configuration object
            if not isinstance(configuration, Configuration):
                raise TypeError(
                    f"config must be a Configuration object, got {type(configuration)}"
                )
            self.cfg = configuration
        elif path_to_config is not None:
            self.cfg = load_configuration(path_to_config)
        else:
            raise ValueError(
                "No configuration or path provided. One of configuration "
                "object, configuration path or model path must be provided."
            )

        # TODO: load checkpoint
        self.model = CAREamicsModel(
            
        )

        # TODO add callbacks here?
        self.trainer = Trainer(max_epochs=self.cfg.training.num_epochs)


    # how to do WandB
    # TODO: how to do AMP? How to continue training? How to load model from checkpoint?
    # TODO: how to save checkpoints?
    # TODO configure training parameters (epochs, etc.), potentially needs to be possible here
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        # TODO sanity check
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def train_on_path(
        self,
        path_to_train_data: Union[Path, str],
        path_to_val_data: Optional[Union[Path, str]] = None,
    ) -> None:
        # sanity check on train data
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
        if path_to_val_data is not None:
            # sanity check on val data
            if not path_to_val_data.exists():
                raise FileNotFoundError(
                    f"Data path {path_to_val_data} is incorrect or"
                    f" does not exist."
                )
            elif not path_to_val_data.is_dir():
                raise ValueError(
                    f"Data path {path_to_val_data} is not a directory."
                )
        # TODO how to deal with no validation data?

        # create datasets and dataloaders
        train_dataset = get_train_dataset(self.cfg, path_to_train_data)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
        )

        val_dataset = get_validation_dataset(self.cfg, path_to_val_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
        )

        # train
        self.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)


    def train_on_array(
        self,
        array_train: np.ndarray,
        array_val: Optional[np.ndarray] = None,
    ) -> None:
        # TODO sanity check
        # TODO create dataloader
        # TODO call self.train
        pass

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
            self.cfg, 
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


    # TODO this method is very similar to predict_on_path, should we keep both?
    def predict_on_array(
        self,
        array: np.ndarray,
        tile_shape: Optional[tuple] = None,
        overlaps: Optional[tuple] = None,
    ) -> Dict[str, np.ndarray]:

        # sanity check
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"Array must be a numpy array, got {type(array)}"
            )   

        # create dataset
        pred_dataset = get_prediction_dataset(
            self.cfg, 
            array,
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
        # TODO save as modelzoo
        pass

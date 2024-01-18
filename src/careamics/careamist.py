from pathlib import Path
from typing import Optional, Union

import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from careamics.config import Configuration, load_configuration
from careamics.lightning import LUNet
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
        self.model = LUNet(self.cfg)

        # TODO trainer in the train function
        self.trainer = Trainer(
            inference_mode=False, max_epochs=2
        )

    # TODO: how to do AMP? How to continue training? How to load model from checkpoint?
    # TODO: how to save checkpoints?
    # TODO configure training parameters (scheduler, etc.)
    def train(
        self,
        *,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        # TODO sanity check on configuration?
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def train_on_path(
        self,
        *,
        path_to_train_data: Union[Path, str],
        path_to_val_data: Optional[Union[Path, str]] = None,
    ) -> None:
        train_dataset = get_train_dataset(self.cfg, path_to_train_data)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=0,
            pin_memory=False,
        )

        val_dataset = get_validation_dataset(self.cfg, path_to_val_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=0,
            pin_memory=False,
        )

        self.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)


    def train_on_array(
        self,
        *,
        array_train: np.ndarray,
        array_val: Optional[np.ndarray] = None,
    ) -> None:
        # TODO create dataloader
        # TODO call self.train
        pass

    def predict(
        self,
        *,
        test_dataloader: Optional[DataLoader] = None,
    ):
        # TODO checks on input
        self.trainer.predict(self.model, test_dataloader)

        # TODO reassemble outputs by calling function
        # call stitch_prediction <- our dataloader, what happens if it is another?

    def predict_on_path(
        self,
        *,
        path_to_test_data: Union[Path, str],
        tile_shape: Optional[tuple] = None,
        overlap: Optional[tuple] = None,
    ):
        # TODO create dataloader
        # TODO call self.predict
        pass

    def predict_on_array(
        self,
        *,
        array: np.ndarray,
        tile_shape: Optional[tuple] = None,
        overlap: Optional[tuple] = None,
    ):
        # TODO create dataloader
        # TODO call self.predict
        pass

    def save(
        self,
        format: str = "modelzoo",  # TODO Enum
    ):
        # TODO save as modelzoo
        pass

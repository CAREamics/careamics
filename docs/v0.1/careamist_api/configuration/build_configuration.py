#!/usr/bin/env python

# %%
# --8<-- [start:pydantic]
from careamics.config import (  # (1)!
    Configuration,
    DataConfig,
    N2VAlgorithm,
    TrainingConfig,
)
from careamics.config.architectures import UNetConfig
from careamics.config.lightning.callbacks.callback_config import (
    CheckpointConfig,
    EarlyStoppingConfig,
)
from careamics.config.support import (
    SupportedData,
    SupportedLogger,
)
from careamics.config.transformations import XYFlipConfig

experiment_name = "N2V_example"

# build the model and algorithm configurations
model = UNetConfig(
    architecture="UNet",  # (2)!
    num_channels_init=64,  # (3)!
    depth=3,
    # (4)!
)

algorithm_model = N2VAlgorithm(  # (5)!
    model=model,
    # (6)!
)

# then the data configuration for N2V
data_model = DataConfig(  # (7)!
    data_type=SupportedData.ARRAY.value,
    patch_size=(256, 256),
    batch_size=8,
    axes="YX",
    transforms=[XYFlipConfig(flip_y=False)],  # (8)!
    dataloader_params={  # (9)!
        "num_workers": 4,
        "shuffle": True,
    },
)

# then the TrainingConfig
earlystopping = EarlyStoppingConfig(
    # (10)!
)

checkpoints = CheckpointConfig(every_n_epochs=10)  # (11)!

training_model = TrainingConfig(
    num_epochs=30,
    logger=SupportedLogger.WANDB.value,
    early_stopping_callback=earlystopping,
    checkpoint_callback=checkpoints,
    # (12)!
)

# finally, build the Configuration
config = Configuration(  # (13)!
    experiment_name=experiment_name,
    algorithm_config=algorithm_model,
    data_config=data_model,
    training_config=training_model,
)
# --8<-- [end:pydantic]

# %%
# --8<-- [start:as_dict]
from careamics.config import Configuration

config_dict = {
    "experiment_name": "N2V_example",
    "algorithm_config": {
        "algorithm": "n2v",  # (1)!
        "loss": "n2v",
        "model": {
            "architecture": "UNet",  # (2)!
            "num_channels_init": 64,
            "depth": 3,
        },
        # (3)!
    },
    "data_config": {
        "data_type": "array",
        "patch_size": [256, 256],
        "batch_size": 8,
        "axes": "YX",
        "transforms": [
            {
                "name": "XYFlip",
                "flip_y": False,
            },
        ],
        "dataloader_params": {
            "num_workers": 4,
        },
    },
    "training_config": {
        "num_epochs": 30,
        "logger": "wandb",
        "early_stopping_callback": {},  # (4)!
        "checkpoint_callback": {
            "every_n_epochs": 10,
        },
    },
}

# instantiate specific configuration
config_as_dict = Configuration(**config_dict)  # (5)!

# --8<-- [end:as_dict]

if config != config_as_dict:
    raise ValueError("Configurations are not equal (Pydantic vs Dict).")

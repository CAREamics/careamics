#!/usr/bin/env python


# --8<-- [start:lightning_api_n2n]
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics.config import create_advanced_n2n_config
from careamics.lightning import (
    CareamicsDataModule,
    CAREModule,
    ConfigSaverCallback,
    convert_prediction,
)

rng = np.random.default_rng(42)
# for this example, we generate fake data
clean_train = rng.uniform(0, 1, size=(12, 64, 64)).astype(np.float32)
train_data = clean_train + rng.normal(0, 0.05, clean_train.shape).astype(np.float32)
train_target = clean_train + rng.normal(0, 0.05, clean_train.shape).astype(np.float32)
clean_val = rng.uniform(0, 1, size=(4, 64, 64)).astype(np.float32)
val_data = clean_val + rng.normal(0, 0.05, clean_val.shape).astype(np.float32)
val_target = clean_val + rng.normal(0, 0.05, clean_val.shape).astype(np.float32)

# create a configuration
config = create_advanced_n2n_config(  # (1)!
    experiment_name="n2n_lightning",
    data_type="array",
    axes="SYX",
    patch_size=(64, 64),
    batch_size=4,
    num_epochs=30,
)
# --8<-- [end:lightning_api_n2n]

config.training_config.trainer_params["max_epochs"] = 2
config.training_config.trainer_params["limit_train_batches"] = 3

# --8<-- [start:lightning_api_n2n_2]
# Lightning DataModule and Module
data_module = CareamicsDataModule(  # (2)!
    data_config=config.data_config,
    train_data=train_data,
    train_data_target=train_target,
    val_data=val_data,
    val_data_target=val_target,
)
model = CAREModule(config.algorithm_config)  # (3)!

# Callbacks
callbacks = [
    ModelCheckpoint(  # (4)!
        dirpath="checkpoints",
        filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
        **config.training_config.checkpoint_params,
    ),
    ConfigSaverCallback(  # (5)!
        config.version,
        config.experiment_name,
        config.training_config,
        config.data_config,
    ),
]

# Lightning Trainer
trainer = Trainer(callbacks=callbacks, **config.training_config.trainer_params)  # (6)!
trainer.fit(model, datamodule=data_module)  # (7)!

# create an inference data config
pred_config = config.data_config.convert_mode(  # (8)!
    new_mode="predicting",
    new_patch_size=(256, 256),
    overlap_size=(48, 48),
    new_batch_size=1,
)

inf_data_module = CareamicsDataModule(  # (9)!
    data_config=pred_config,
    pred_data=train_data,
)

# run inference
tiled_predictions = trainer.predict(model, datamodule=inf_data_module)  # (10)!

# convert list of tile predictions to stitched data
stitched_predictions, sources = convert_prediction(  # (11)!
    tiled_predictions,
    tiled=True,
)
# --8<-- [end:lightning_api_n2n_2]

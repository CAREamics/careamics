#!/usr/bin/env python

# --8<-- [start:lightning_api_n2v]
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import pooch

from careamics.config import create_advanced_n2v_config
from careamics.lightning import (
    CareamicsDataModule,
    N2VModule,
    ConfigSaverCallback,
    convert_prediction,
)

# download example data
train_image = pooch.retrieve(
    "https://zenodo.org/records/21028053/files/train.tif",
    known_hash="8be263564a12381bcc0fc69c4271728f3a794aeed78ef85be5fac95e78f5ff73",
)
val_image = pooch.retrieve(
    "https://zenodo.org/records/21028053/files/validation.tif",
    known_hash="6f5cd80d4e7f086432458987ee09c7623b2f0c98568bdf2f05b747d804abfabb",
)

# create configuration
config = create_advanced_n2v_config(  # (1)!
    experiment_name="n2v_lightning",
    data_type="tiff",
    axes="YX",
    patch_size=(64, 64),
    batch_size=4,
    num_epochs=30,
)
# --8<-- [end:lightning_api_n2v]

config.training_config.trainer_params["max_epochs"] = 2
config.training_config.trainer_params["limit_train_batches"] = 3

# --8<-- [start:lightning_api_n2v_2]
# Lightning DataModule and Module
data_module = CareamicsDataModule(  # (2)!
    data_config=config.data_config,
    train_data=train_image,
    val_data=val_image,
)
model = N2VModule(config.algorithm_config)  # (3)!

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
    pred_data=train_image,
)

# run inference
tiled_predictions = trainer.predict(model, datamodule=inf_data_module)  # (10)!

# convert list of tile predictions to stitched data
stitched_predictions, sources = convert_prediction(  # (11)!
    tiled_predictions,
    tiled=True,
)

# --8<-- [end:lightning_api_n2v_2]

# --8<-- [start:predict_to_disk]
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import pooch

from careamics.config import create_advanced_n2v_config
from careamics.lightning import (
    CareamicsDataModule,
    N2VModule,
    ConfigSaverCallback,
    convert_prediction,
)
from careamics.lightning.callbacks.prediction import PredictionWriterCallback
from careamics.lightning.callbacks.prediction import (
    PredictionWriterCallback,
)

pred_writer = PredictionWriterCallback(
    dirpath="predictions", enable_writing=False  # (1)!
)

callbacks = [
    ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
        **config.training_config.checkpoint_params,
    ),
    ConfigSaverCallback(
        config.version,
        config.experiment_name,
        config.training_config,
        config.data_config,
    ),
    pred_writer,  # (2)!
]

trainer = Trainer(
    enable_progress_bar=True,
    callbacks=callbacks,
    **config.training_config.trainer_params,
)

trainer.fit(model, datamodule=data_module)

# create an inference data config
pred_config = config.data_config.convert_mode(
    new_mode="predicting",
    new_patch_size=(256, 256),
    overlap_size=(48, 48),
    new_batch_size=1,
)

inf_data_module = CareamicsDataModule(
    data_config=pred_config,
    pred_data=train_image,
)

# run inference with predictions written to disk
pred_writer.set_writing_strategy("tiff", tiled=True)  # (3)!
pred_writer.enable_writing(True)  # (4)!

_ = trainer.predict(
    model,
    datamodule=inf_data_module,
    return_predictions=False,  # (5)!
)
# --8<-- [end:predict_to_disk]

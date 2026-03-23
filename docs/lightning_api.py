#!/usr/bin/env python

# %%
# --8<-- [start:lightning_api]
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from careamics_portfolio import PortfolioManager

from careamics.config.ng_factories import create_advanced_n2v_config
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import N2VModule
from careamics.lightning.dataset_ng.prediction import convert_prediction
from careamics.lightning.callbacks import CareamicsCheckpointInfo

# download example data
portfolio_manager = PortfolioManager()
files = portfolio_manager.denoising.N2V_SEM.download()
train_image = files[0]
val_image = files[1]

# create configuration
config = create_advanced_n2v_config(  # (1)!
    experiment_name="na",  # unused in LightningAPI
    data_type="tiff",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=1,
    num_workers=0,  # (2)!
)

# create lightning modules
model = N2VModule(config.algorithm_config)  # (3)!

data_module = CareamicsDataModule(  # (4)!
    data_config=config.data_config,
    train_data=train_image,
    val_data=val_image,
)

callbacks = [
    ModelCheckpoint(  # (5)!
        dirpath="checkpoints",
        filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
        **config.training_config.checkpoint_callback.model_dump(),
    ),
    CareamicsCheckpointInfo(  # (6)!
        config.version, config.experiment_name, config.training_config
    ),
]

trainer = Trainer(
    enable_progress_bar=True,
    callbacks=callbacks,
    **config.training_config.lightning_trainer_config,  # (7)!
)

trainer.fit(model, datamodule=data_module)  # (8)!

# create an inference data config
pred_config = config.data_config.convert_mode(  # (9)!
    new_mode="predicting",
    new_patch_size=(256, 256),
    overlap_size=(48, 48),
    new_batch_size=1,
)

inf_data_module = CareamicsDataModule(  # (10)!
    data_config=pred_config,
    pred_data=train_image,
)

# run inference
tiled_predictions = trainer.predict(model, datamodule=inf_data_module)  # (11)!

# convert list of tile predictions to stitched data
stitched_predictions, sources = convert_prediction(  # (12)!
    tiled_predictions,
    tiled=True,
)
# --8<-- [end:lightning_api]

# --8<-- [start:predict_to_disk]
from careamics.lightning.dataset_ng.callbacks.prediction_writer import (
    PredictionWriterCallback,
)

pred_writer = PredictionWriterCallback(  # (1)!
    dirpath="predictions", enable_writing=False
)

callbacks = [
    ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
        **config.training_config.checkpoint_callback.model_dump(),
    ),
    CareamicsCheckpointInfo(
        config.version, config.experiment_name, config.training_config
    ),
    pred_writer,  # (2)!
]

trainer = Trainer(
    enable_progress_bar=True,
    callbacks=callbacks,
    **config.training_config.lightning_trainer_config,
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

# run inference
pred_writer.set_writing_strategy("tiff", tiled=True)  # (3)!
pred_writer.enable_writing(True)  # (4)!

tiled_predictions = trainer.predict(
    model, datamodule=inf_data_module, return_predictions=False  # (5)!
)
# --8<-- [end:predict_to_disk]
from pathlib import Path

assert (Path("predictions") / "train.tiff").exists()
# assert (Path("predictions") / train )

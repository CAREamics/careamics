#!/usr/bin/env python
# %%
# --8<-- [start:lightning_api_n2v]
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import pooch

from careamics.config import create_advanced_n2v_config
from careamics.lightning import (
    CareamicsDataModule,
    ConfigSaverCallback,
    N2VModule,
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
        **config.training_config.checkpoint_params,
    ),
    ConfigSaverCallback(
        config.version,
        config.experiment_name,
        config.training_config,
        config.data_config,
    ),  # (6)!
]

trainer = Trainer(
    enable_progress_bar=True,
    callbacks=callbacks,
    **config.training_config.trainer_params,  # (7)!
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
# --8<-- [end:lightning_api_n2v]

# --8<-- [start:lightning_api_care]
from careamics.config import create_advanced_care_config
from careamics.lightning.modules import CAREModule

config = create_advanced_care_config(
    experiment_name="care_lightning",
    data_type="tiff",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=1,
    num_workers=0,
)
model = CAREModule(config.algorithm_config)
data_module = CareamicsDataModule(
    data_config=config.data_config,
    train_data=train_image,
    train_data_target=val_image,
    val_data=val_image,
    val_data_target=val_image,
)
trainer = Trainer(**config.training_config.trainer_params)
trainer.fit(model, datamodule=data_module)
# --8<-- [end:lightning_api_care]

# --8<-- [start:lightning_api_n2n]
from careamics.config import create_advanced_n2n_config
from careamics.lightning.modules import CAREModule

config = create_advanced_n2n_config(
    experiment_name="n2n_lightning",
    data_type="tiff",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=1,
    num_workers=0,
)
model = CAREModule(config.algorithm_config)
data_module = CareamicsDataModule(
    data_config=config.data_config,
    train_data=train_image,
    train_data_target=val_image,
    val_data=val_image,
    val_data_target=val_image,
)
trainer = Trainer(**config.training_config.trainer_params)
trainer.fit(model, datamodule=data_module)
# --8<-- [end:lightning_api_n2n]

# --8<-- [start:lightning_api_hdn]
from careamics.config import create_advanced_hdn_config
from careamics.lightning.modules import HDNModule

config = create_advanced_hdn_config(
    experiment_name="hdn_lightning",
    data_type="tiff",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    use_noise_model=True,
    num_epochs=1,
    num_workers=0,
)
model = HDNModule(config.algorithm_config)
# Load one trained noise model for each model output channel.
model.set_noise_model(["noise_models/noise_model_ch0.npz"])
data_module = CareamicsDataModule(
    data_config=config.data_config,
    train_data=train_image,
    val_data=val_image,
)
trainer = Trainer(**config.training_config.trainer_params)
trainer.fit(model, datamodule=data_module)
# --8<-- [end:lightning_api_hdn]

# --8<-- [start:lightning_api_microsplit]
from careamics.config import create_advanced_microsplit_config
from careamics.lightning.modules import MicroSplitModule

config = create_advanced_microsplit_config(
    experiment_name="microsplit_lightning",
    data_type="tiff",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    output_channels=1,
    noise_model_likelihood_weight=0.9,
    num_epochs=1,
    num_workers=0,
)
model = MicroSplitModule(config.algorithm_config)
# Load one trained noise model for each model output channel.
model.set_noise_model(["noise_models/noise_model_ch0.npz"])
data_module = CareamicsDataModule(
    data_config=config.data_config,
    train_data=train_image,
    train_data_target=train_image,
    val_data=val_image,
    val_data_target=val_image,
)
trainer = Trainer(**config.training_config.trainer_params)
trainer.fit(model, datamodule=data_module)
# --8<-- [end:lightning_api_microsplit]

# --8<-- [start:lightning_api_cryocare]
from careamics.config import create_advanced_cryocare_config
from careamics.lightning.modules import CryoCAREModule

config = create_advanced_cryocare_config(
    experiment_name="cryocare_lightning",
    data_type="mrc",
    axes="ZYX",
    patch_size=(16, 64, 64),
    batch_size=8,
    num_epochs=1,
    num_workers=0,
)
model = CryoCAREModule(config.algorithm_config)
data_module = CareamicsDataModule(
    data_config=config.data_config,
    train_data=train_image,
    train_data_target=val_image,
    val_data=val_image,
    val_data_target=train_image,
)
trainer = Trainer(**config.training_config.trainer_params)
trainer.fit(model, datamodule=data_module)
# --8<-- [end:lightning_api_cryocare]

# --8<-- [start:lightning_api_segmentation]
from careamics.config import create_advanced_seg_config
from careamics.lightning.modules import SegModule

config = create_advanced_seg_config(
    experiment_name="segmentation_lightning",
    data_type="tiff",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    n_classes=2,
    num_epochs=1,
    num_workers=0,
)
model = SegModule(config.algorithm_config)
data_module = CareamicsDataModule(
    data_config=config.data_config,
    train_data=train_image,
    train_data_target=train_image,
    val_data=val_image,
    val_data_target=val_image,
)
trainer = Trainer(**config.training_config.trainer_params)
trainer.fit(model, datamodule=data_module)
# --8<-- [end:lightning_api_segmentation]

# --8<-- [start:predict_to_disk]
from careamics.lightning.callbacks.prediction import (
    PredictionWriterCallback,
)

pred_writer = PredictionWriterCallback(
    dirpath="predictions", enable_writing=False
)  # (1)!

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

import shutil

# delete outputs
shutil.rmtree("predictions")

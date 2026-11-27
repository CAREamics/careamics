#!/usr/bin/env python


# --8<-- [start:lightning_api_hdn]
from pathlib import Path

import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics.config import create_advanced_hdn_config
from careamics.lightning import CareamicsDataModule, ConfigSaverCallback
from careamics.lightning.modules import HDNModule
from careamics.lightning.prediction import convert_predict_outputs
from careamics.noise_model import train_noise_model

# for this example, we generate fake data
rng = np.random.default_rng(42)
signal = rng.uniform(0, 255, size=(12, 64, 64)).astype(np.float32)
observation = signal + rng.normal(0, 10, size=signal.shape).astype(np.float32)
val_signal = rng.uniform(0, 255, size=(4, 64, 64)).astype(np.float32)
val_observation = val_signal + rng.normal(0, 10, size=val_signal.shape).astype(
    np.float32
)

# we may need a noise model
noise_model_dir = Path("noise_models_hdn")
train_noise_model(  # (1)!
    signal=signal,
    observation=observation,
    axes="SYX",
    n_gaussian=1,
    n_coeff=2,
    n_epochs=10,
    save_dir=noise_model_dir,
)

# create a configuration
config = create_advanced_hdn_config(  # (2)!
    experiment_name="hdn_lightning",
    data_type="array",
    axes="SYX",
    patch_size=(64, 64),
    batch_size=4,
    num_epochs=30,
    use_noise_model=True,
)
# --8<-- [end:lightning_api_hdn]

config.training_config.trainer_params["max_epochs"] = 2
config.training_config.trainer_params["limit_train_batches"] = 3

# --8<-- [start:lightning_api_hdn_2]
# Lightning DataModule and Module
data_module = CareamicsDataModule(  # (3)!
    data_config=config.data_config,
    train_data=observation,
    val_data=val_observation,
)
model = HDNModule(config.algorithm_config)  # (4)!
model.set_noise_model(sorted(noise_model_dir.glob("*.npz")))  # (5)!

# Callbacks
callbacks = [
    ModelCheckpoint(  # (6)!
        dirpath="checkpoints",
        filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
        **config.training_config.checkpoint_params,
    ),
    ConfigSaverCallback(  # (7)!
        config.version,
        config.experiment_name,
        config.training_config,
        config.data_config,
    ),
]

# Lightning Trainer
trainer = Trainer(callbacks=callbacks, **config.training_config.trainer_params)  # (8)!
trainer.fit(model, datamodule=data_module)  # (9)!

# create an inference data config
pred_config = config.data_config.convert_mode(  # (10)!
    new_mode="predicting",
    new_patch_size=(256, 256),
    overlap_size=(48, 48),
    new_batch_size=1,
)

inf_data_module = CareamicsDataModule(  # (11)!
    data_config=pred_config,
    pred_data=observation,
)

# run inference
tiled_predictions = trainer.predict(model, datamodule=inf_data_module)  # (12)!

# convert list of tile predictions to stitched data
stitched_predictions, uncertainties, sources = convert_predict_outputs(  # (13)!
    tiled_predictions,
    tiled=True,
)
# --8<-- [end:lightning_api_hdn_2]

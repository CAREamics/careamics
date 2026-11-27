#!/usr/bin/env python

from pathlib import Path

import numpy as np

train_data = np.random.randint(0, 255, (4, 1, 64 * 4, 64 * 8)).astype(np.float32)
train_data_target = np.repeat(train_data, 3, axis=1)
val_data = train_data
val_data_target = train_data_target
pred_data = train_data

noise_model = []
for channel in range(3):
    noise_model_path = Path(f"dummy_noise_model_ch{channel}.npz")
    np.savez(
        noise_model_path,
        trained_weight=np.ones((9, 3), dtype=np.float32),
        min_signal=np.array([0.0]),
        max_signal=np.array([255.0]),
        min_sigma=np.array([0.125]),
    )
    noise_model.append(noise_model_path)


# --8<-- [start:quick_start_microsplit_config]
from careamics import CAREamist
from careamics.config import create_microsplit_config

# Create a configuration
config = create_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",  # (1)!
    axes="SCYX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,  # (2)!
    num_epochs=50,
)
# --8<-- [end:quick_start_microsplit_config]

config.training_config.trainer_params["max_epochs"] = 2
config.training_config.trainer_params["limit_train_batches"] = 3

# --8<-- [start:quick_start_microsplit_train]
# Instantiate a CAREamist from the config
careamist = CAREamist(config)

# Train  using a noise model
careamist.train(
    train_data=train_data,  # (4)!
    train_data_target=train_data_target,
    val_data=train_data,
    val_data_target=train_data_target,
    noise_model=noise_model,  # (5)!
)

# Predict on the data of choice
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_microsplit_train]

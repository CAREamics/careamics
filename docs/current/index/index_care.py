#!/usr/bin/env python

import numpy as np

train_data = np.random.randint(0, 255, (64 * 4, 64 * 8)).astype(np.float32)
train_target = train_data
val_data = train_data
val_target = train_data
pred_data = train_data


# --8<-- [start:quick_start_care_config]
from careamics import CAREamist
from careamics.config import create_care_config

# Create a configuration
config = create_care_config(
    experiment_name="care_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=50,
)
# --8<-- [end:quick_start_care_config]

config.training_config.trainer_params["max_epochs"] = 2
config.training_config.trainer_params["limit_train_batches"] = 3

# --8<-- [start:quick_start_care_train]
# Instantiate a CAREamist from the config
careamist = CAREamist(config)

# Train the model
careamist.train(
    train_data=train_data,  # (2)!
    train_data_target=train_target,  # (3)!
    val_data=val_data,
    val_data_target=val_target,
)

# Predict on the data of choice
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_care_train]

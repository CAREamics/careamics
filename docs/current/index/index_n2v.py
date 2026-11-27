#!/usr/bin/env python

import numpy as np

train_data = np.random.randint(0, 255, (64 * 4, 64 * 8)).astype(np.float32)
pred_data = train_data


# --8<-- [start:quick_start_n2v_config]
from careamics import CAREamist
from careamics.config import create_n2v_config

# Create a configuration
config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=50,
)
# --8<-- [end:quick_start_n2v_config]

config.training_config.trainer_params["max_epochs"] = 2
config.training_config.trainer_params["limit_train_batches"] = 3

# --8<-- [start:quick_start_n2v_train]
# Instantiate a CAREamist from the config
careamist = CAREamist(config)

# Train the model
careamist.train(train_data=train_data)  # (2)!

# Predict on the data of choice
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_n2v_train]

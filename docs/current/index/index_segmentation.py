#!/usr/bin/env python

import numpy as np

train_data = np.random.randint(0, 255, (64 * 4, 64 * 8)).astype(np.float32)
train_labels = np.random.randint(0, 2, train_data.shape).astype(np.int64)


# --8<-- [start:quick_start_segmentation_config]
from careamics import CAREamist
from careamics.config import create_seg_config

# Create a configuration
config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=50,
    n_classes=2,  # (2)!
    prediction_output="probabilities",  # (3)!
)
# --8<-- [end:quick_start_segmentation_config]

config.training_config.trainer_params["max_epochs"] = 2
config.training_config.trainer_params["limit_train_batches"] = 3

# --8<-- [start:quick_start_segmentation_train]
# Instantiate a CAREamist from the config
careamist = CAREamist(config)

# Train the model
careamist.train(
    train_data=train_data,  # (4)!
    train_data_target=train_labels,  # (5)!
)

# Predict on the data of choice
predictions, sources = careamist.predict(pred_data=train_data)
# --8<-- [end:quick_start_segmentation_train]

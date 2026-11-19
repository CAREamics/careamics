#!/usr/bin/env python

import numpy as np

train_data = np.random.randint(0, 255, (64 * 4, 64 * 8)).astype(np.float32)
train_labels = np.random.randint(0, 2, train_data.shape).astype(np.int64)


# --8<-- [start:quick_start_segmentation_config]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,  # (2)!
    prediction_output="probabilities",  # (3)!
    num_epochs=2,
    num_steps=5,
)
# --8<-- [end:quick_start_segmentation_config]

# --8<-- [start:quick_start_segmentation_train]
from careamics.careamist import CAREamist

careamist = CAREamist(config)
careamist.train(
    train_data=train_data,  # (4)!
    train_data_target=train_labels,  # (5)!
)
predictions, sources = careamist.predict(pred_data=train_data)
# --8<-- [end:quick_start_segmentation_train]

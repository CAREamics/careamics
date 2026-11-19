#!/usr/bin/env python

import numpy as np

even_reconstruction = np.random.randint(0, 255, (16, 64, 64)).astype(np.float32)
odd_reconstruction = even_reconstruction.copy()


# --8<-- [start:quick_start_cryocare_config]
from careamics.config import create_cryocare_config

config = create_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",  # (1)!
    axes="ZYX",  # (2)!
    patch_size=[16, 64, 64],
    batch_size=8,
    tilt_axis="Y",  # (3)!
    num_epochs=2,
    num_steps=5,
)
# --8<-- [end:quick_start_cryocare_config]

# --8<-- [start:quick_start_cryocare_train]
from careamics.careamist import CAREamist

careamist = CAREamist(config)
careamist.train(
    train_data=even_reconstruction,  # (4)!
    train_data_target=odd_reconstruction,  # (5)!
)
careamist.predict_to_disk(
    pred_data=even_reconstruction,  # (6)!
    pred_data_target=odd_reconstruction,  # (7)!
    prediction_dir="cryocare_predictions",
)
# --8<-- [end:quick_start_cryocare_train]

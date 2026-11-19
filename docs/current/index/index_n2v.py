#!/usr/bin/env python

import numpy as np

train_data = np.random.randint(0, 255, (64 * 4, 64 * 8)).astype(np.float32)
pred_data = train_data


# --8<-- [start:quick_start_n2v_config]
from careamics.config import create_n2v_config

config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=2,
    num_steps=5,
)
# --8<-- [end:quick_start_n2v_config]

# --8<-- [start:quick_start_n2v_train]
from careamics.careamist import CAREamist

careamist = CAREamist(config)
careamist.train(train_data=train_data)  # (2)!
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_n2v_train]

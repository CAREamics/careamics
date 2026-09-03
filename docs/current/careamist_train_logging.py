#!/usr/bin/env python
import numpy as np
from careamics.careamist import CAREamist
from careamics.config.factories import create_n2v_config

rng = np.random.default_rng(seed=0)
train_data = rng.integers(0, 255, (512, 512)).astype(np.float32)

config = create_n2v_config(
    experiment_name="n2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=2,
)
careamist = CAREamist(config)
careamist.train(train_data=train_data)

# %%
# --8<-- [start:csv_logger]
from careamics.plotting import plot_loss

training_report = careamist.get_losses()  # (1)!
plot_loss(training_report)  # (2)!

# --8<-- [end:csv_logger]

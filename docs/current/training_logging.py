#!/usr/bin/env python
import numpy as np
from careamics.config.factories import create_n2v_config
from careamics.careamist import CAREamist

train_data = np.random.randint(0, 255, (512, 512)).astype(np.float32)

config_n2v = create_n2v_config(
    experiment_name="n2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=2,
)
careamist = CAREamist(config_n2v)

careamist.train(train_data=train_data)

# %%
# --8<-- [start:csv_logger]
import matplotlib.pyplot as plt

losses = careamist.get_losses()

plt.plot(
    losses["train_epoch"], losses["train_loss"], losses["val_epoch"], losses["val_loss"]
)
plt.legend(["Train loss", "Val loss"])
plt.title("Losses")
# --8<-- [end:csv_logger]

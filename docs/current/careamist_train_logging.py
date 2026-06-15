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
import matplotlib.pyplot as plt

losses = careamist.get_losses()  # (1)!

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(losses["train_epoch"], losses["train_loss"], label="Train")
ax.plot(losses["val_epoch"], losses["val_loss"], label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(frameon=False)
fig.tight_layout()
# --8<-- [end:csv_logger]

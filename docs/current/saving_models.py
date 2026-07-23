#!/usr/bin/env python
from pathlib import Path

import numpy as np

from careamics.careamist import CAREamist
from careamics.config.factories import create_n2v_config

root = Path(__file__).parent / "temp_data"
root.mkdir(exist_ok=True)

# create a configuration and train a small model
config = create_n2v_config(
    experiment_name="saving_models",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
    num_steps=1,
    n_val_patches=2,
)

train_data = np.random.randint(0, 255, (512, 512)).astype(np.float32)

careamist = CAREamist(config, work_dir=root)
careamist.train(train_data=train_data)


# %%
# --8<-- [start:get_checkpoints]
checkpoints = careamist.get_checkpoints()
# --8<-- [end:get_checkpoints]

# %%
# --8<-- [start:reload]
best_careamist = CAREamist(checkpoint_path=checkpoints[0])
# --8<-- [end:reload]

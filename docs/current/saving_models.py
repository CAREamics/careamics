#!/usr/bin/env python
from pathlib import Path

import numpy as np

from careamics.careamist import CAREamist
from careamics.config import create_care_config

root = Path(__file__).parent / "temp_data"
root.mkdir(exist_ok=True)

# create a configuration and train a small model
# Using CARE so that the top-k checkpoints will be saved.
config = create_care_config(
    experiment_name="saving_models",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=2,
    num_steps=1,
    n_val_patches=2,
)

train_data = np.random.randint(0, 255, (512, 512)).astype(np.float32)

careamist = CAREamist(config, work_dir=root)
careamist.train(train_data=train_data, train_data_target=train_data)


# %%
# --8<-- [start:get_checkpoints]
checkpoint_record = careamist.get_checkpoints()
if checkpoint_record is not None:
    top_k = checkpoint_record["best_k"]
# --8<-- [end:get_checkpoints]

# %%
# --8<-- [start:reload]
checkpoint_dir = careamist.get_checkpoint_root()
# load the second best checkpoint
reloaded_careamist = CAREamist(
    checkpoint_path=checkpoint_dir / top_k[1]["name"],
)
# --8<-- [end:reload]

# %%
# --8<-- [start:util-load]
from careamics.utils import read_checkpoint_record

checkpoint_record = read_checkpoint_record(checkpoint_dir)
# --8<-- [end:util-load]

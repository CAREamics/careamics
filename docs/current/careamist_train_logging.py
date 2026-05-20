#!/usr/bin/env python
import os

# Run WandB offline so the snippets work in CI and without a logged-in account.
# Override by exporting WANDB_MODE=online (or =disabled) in the shell.
os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np
from careamics.careamist import CAREamist
from careamics.config.factories import create_advanced_n2v_config, create_n2v_config

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

# %%
# --8<-- [start:wandb_env]
import os

os.environ["WANDB_MODE"] = "offline"  # (1)!
os.environ["WANDB_PROJECT"] = "careamics-experiments"  # (2)!
os.environ["WANDB_ENTITY"] = "your-team"  # (3)!
# --8<-- [end:wandb_env]

# %%
# --8<-- [start:wandb_enable]
from careamics.careamist import CAREamist
from careamics.config.factories import create_advanced_n2v_config

config = create_advanced_n2v_config(
    experiment_name="n2v_wandb",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=2,
    logger="wandb",  # (1)!
)
careamist = CAREamist(config)
careamist.train(train_data=train_data)
# --8<-- [end:wandb_enable]

# %%
# --8<-- [start:tensorboard_enable]
from careamics.careamist import CAREamist
from careamics.config.factories import create_advanced_n2v_config

config = create_advanced_n2v_config(
    experiment_name="n2v_tb",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=2,
    logger="tensorboard",  # (1)!
)
careamist = CAREamist(config)
careamist.train(train_data=train_data)
# --8<-- [end:tensorboard_enable]

# %%
# --8<-- [start:compare_experiments]
from pathlib import Path

from careamics.careamist import CAREamist
from careamics.config.factories import create_advanced_n2v_config

experiments = [
    {"name": "baseline", "patch_size": [64, 64], "batch_size": 8},
    {"name": "large_patches", "patch_size": [128, 128], "batch_size": 4},
]

base_dir = Path("tb_comparison")
for exp in experiments:
    config = create_advanced_n2v_config(
        experiment_name=exp["name"],
        data_type="array",
        axes="YX",
        patch_size=exp["patch_size"],
        batch_size=exp["batch_size"],
        num_epochs=2,
        logger="tensorboard",
    )
    careamist = CAREamist(config, work_dir=base_dir / exp["name"])  # (1)!
    careamist.train(train_data=train_data)
# --8<-- [end:compare_experiments]

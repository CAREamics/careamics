#!/usr/bin/env python

################# cryoCARE #################
# %%
# --8<-- [start:adv_config_cryocare_trainer]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    trainer_params={},  # (1)!
)
# --8<-- [end:adv_config_cryocare_trainer]

# %%
# --8<-- [start:adv_config_cryocare_opt]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    optimizer="Adam",
    optimizer_params={"lr": 4e-4},  # (1)!
)
# --8<-- [end:adv_config_cryocare_opt]

# %%
# --8<-- [start:adv_config_cryocare_lr]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    lr_scheduler="StepLR",
    lr_scheduler_params={"step_size": 10, "gamma": 0.1},  # (1)!
)
# --8<-- [end:adv_config_cryocare_lr]

# %%
# --8<-- [start:adv_config_cryocare_train_dm]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    train_dataloader_params={"shuffle": True, "drop_last": True},
)
# --8<-- [end:adv_config_cryocare_train_dm]

# %%
# --8<-- [start:adv_config_cryocare_checkpoint]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    checkpoint_params={"monitor": "val_loss", "mode": "min", "save_top_k": 3},
)
# --8<-- [end:adv_config_cryocare_checkpoint]

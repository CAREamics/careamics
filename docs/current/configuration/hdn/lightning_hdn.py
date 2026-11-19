#!/usr/bin/env python

################# HDN #################
# %%
# --8<-- [start:adv_config_hdn_model]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    model_params={"z_dims": [128, 128], "n_filters": 32},
)
# --8<-- [end:adv_config_hdn_model]

# %%
# --8<-- [start:adv_config_hdn_trainer]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    trainer_params={},  # (1)!
)
# --8<-- [end:adv_config_hdn_trainer]

# %%
# --8<-- [start:adv_config_hdn_opt]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    optimizer="Adamax",
    optimizer_params={"lr": 3e-4},  # (1)!
)
# --8<-- [end:adv_config_hdn_opt]

# %%
# --8<-- [start:adv_config_hdn_lr]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    lr_scheduler="StepLR",
    lr_scheduler_params={"step_size": 10, "gamma": 0.1},  # (1)!
)
# --8<-- [end:adv_config_hdn_lr]

# %%
# --8<-- [start:adv_config_hdn_train_dm]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    train_dataloader_params={"shuffle": True, "drop_last": True},
)
# --8<-- [end:adv_config_hdn_train_dm]

# %%
# --8<-- [start:adv_config_hdn_checkpoint]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    checkpoint_params={"monitor": "val_loss", "mode": "min", "save_top_k": 3},
)
# --8<-- [end:adv_config_hdn_checkpoint]

# %%
# --8<-- [start:adv_config_hdn_early_stop]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    early_stopping_params={"monitor": "val_loss", "mode": "min", "patience": 5},
)
# --8<-- [end:adv_config_hdn_early_stop]

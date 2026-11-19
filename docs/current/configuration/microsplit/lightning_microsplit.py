#!/usr/bin/env python

################# MicroSplit #################
# %%
# --8<-- [start:adv_config_microsplit_model]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    model_params={"z_dims": [128, 128], "n_filters": 32},
)
# --8<-- [end:adv_config_microsplit_model]

# %%
# --8<-- [start:adv_config_microsplit_trainer]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    trainer_params={},  # (1)!
)
# --8<-- [end:adv_config_microsplit_trainer]

# %%
# --8<-- [start:adv_config_microsplit_opt]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    optimizer="Adamax",
    optimizer_params={"lr": 1e-3},  # (1)!
)
# --8<-- [end:adv_config_microsplit_opt]

# %%
# --8<-- [start:adv_config_microsplit_lr]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    lr_scheduler="StepLR",
    lr_scheduler_params={"step_size": 10, "gamma": 0.1},  # (1)!
)
# --8<-- [end:adv_config_microsplit_lr]

# %%
# --8<-- [start:adv_config_microsplit_train_dm]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    train_dataloader_params={"shuffle": True, "drop_last": True},
)
# --8<-- [end:adv_config_microsplit_train_dm]

# %%
# --8<-- [start:adv_config_microsplit_checkpoint]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    checkpoint_params={"monitor": "val_loss", "mode": "min", "save_top_k": 3},
)
# --8<-- [end:adv_config_microsplit_checkpoint]

# %%
# --8<-- [start:adv_config_microsplit_early_stop]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    early_stopping_params={"monitor": "val_loss", "mode": "min", "patience": 5},
)
# --8<-- [end:adv_config_microsplit_early_stop]

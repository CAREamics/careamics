#!/usr/bin/env python

############################################
################ Noise2Void ################
# %%
# --8<-- [start:adv_config_n2v_trainer]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    trainer_params={},  # (1)!
)
# --8<-- [end:adv_config_n2v_trainer]

# %%
# --8<-- [start:adv_config_n2v_no_val]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_val_patches=0,  # (1)!
    monitor_metric="train_loss_epoch",  # (2)!
)
# --8<-- [end:adv_config_n2v_no_val]

# %%
# --8<-- [start:adv_config_n2v_model]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    model_params={
        "depth": 4,
    },
)
# --8<-- [end:adv_config_n2v_model]

# %%
# --8<-- [start:adv_config_n2v_opt]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    optimizer="Adam",  # (1)!
    optimizer_params={  # (2)!
        "lr": 1e-4,
    },
)
# --8<-- [end:adv_config_n2v_opt]
# %%
# --8<-- [start:adv_config_n2v_lr]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    lr_scheduler="StepLR",  # (1)!
    lr_scheduler_params={  # (2)!
        "step_size": 10,
        "gamma": 0.1,
    },
)
# --8<-- [end:adv_config_n2v_lr]
# %%
# --8<-- [start:adv_config_n2v_train_dm]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    train_dataloader_params={
        "shuffle": True,
        "drop_last": True,
    },
)
# --8<-- [end:adv_config_n2v_train_dm]

# %%
# --8<-- [start:adv_config_n2v_checkpoint]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    checkpoint_params={
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 3,
    },
)
# --8<-- [end:adv_config_n2v_checkpoint]

############################################
################# CARE/N2N #################
# %%
# --8<-- [start:adv_config_care_trainer]
from careamics.config.ng_factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    trainer_params={},  # (1)!
)
# --8<-- [end:adv_config_care_trainer]
# %%
# --8<-- [start:adv_config_care_model]
from careamics.config.ng_factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    model_params={  # (1)!
        "depth": 4,
    },
)
# --8<-- [end:adv_config_care_model]

# %%
# --8<-- [start:adv_config_care_opt]
from careamics.config.ng_factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    optimizer="Adam",  # (1)!
    optimizer_params={  # (2)!
        "lr": 1e-4,
    },
)
# --8<-- [end:adv_config_care_opt]
# %%
# --8<-- [start:adv_config_care_lr]
from careamics.config.ng_factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    lr_scheduler="StepLR",  # (1)!
    lr_scheduler_params={  # (2)!
        "step_size": 10,
        "gamma": 0.1,
    },
)
# --8<-- [end:adv_config_care_lr]
# %%
# --8<-- [start:adv_config_care_train_dm]
from careamics.config.ng_factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    train_dataloader_params={  # (1)!
        "shuffle": True,
        "drop_last": True,
    },
)
# --8<-- [end:adv_config_care_train_dm]
# %%
# --8<-- [start:adv_config_care_checkpoint]
from careamics.config.ng_factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    checkpoint_params={  # (1)!
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 3,
    },
)
# --8<-- [end:adv_config_care_checkpoint]

# %%
# --8<-- [start:adv_config_care_early_stop]
from careamics.config.ng_factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    early_stopping_params={  # (1)!
        "monitor": "val_loss",
        "mode": "min",
        "patience": 5,
    },
)
# --8<-- [end:adv_config_care_early_stop]

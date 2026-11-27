#!/usr/bin/env python

################# Segmentation #################
# %%
# --8<-- [start:adv_config_seg_model]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    model_params={"depth": 3, "num_channels_init": 64, "use_batch_norm": True},
)
# --8<-- [end:adv_config_seg_model]

# %%
# --8<-- [start:adv_config_seg_trainer]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    trainer_params={},  # (1)!
)
# --8<-- [end:adv_config_seg_trainer]

# %%
# --8<-- [start:adv_config_seg_opt]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    optimizer="Adam",
    optimizer_params={"lr": 1e-4},  # (1)!
)
# --8<-- [end:adv_config_seg_opt]

# %%
# --8<-- [start:adv_config_seg_lr]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    lr_scheduler="StepLR",
    lr_scheduler_params={"step_size": 10, "gamma": 0.1},  # (1)!
)
# --8<-- [end:adv_config_seg_lr]

# %%
# --8<-- [start:adv_config_seg_train_dm]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    train_dataloader_params={"shuffle": True, "drop_last": True},
)
# --8<-- [end:adv_config_seg_train_dm]

# %%
# --8<-- [start:adv_config_seg_checkpoint]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    checkpoint_params={"monitor": "val_loss", "mode": "min", "save_top_k": 3},
)
# --8<-- [end:adv_config_seg_checkpoint]

# %%
# --8<-- [start:adv_config_seg_early_stop]
from careamics.config import create_advanced_seg_config

config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    early_stopping_params={"monitor": "val_loss", "mode": "min", "patience": 5},
)
# --8<-- [end:adv_config_seg_early_stop]

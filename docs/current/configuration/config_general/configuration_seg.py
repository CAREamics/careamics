#!/usr/bin/env python

# %%
# --8<-- [start:all_configs]
from careamics.config import create_advanced_seg_config, create_seg_config

# --8<-- [end:all_configs]

# %%
# --8<-- [start:config_seg]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="ZYX",
    patch_size=[16, 64, 64],  # (1)!
    batch_size=8,
    n_classes=2,  # (2)!
    num_epochs=30,
)
# --8<-- [end:config_seg]

# %%
# --8<-- [start:config_seg_steps]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    num_epochs=30,
    num_steps=500,  # (1)!
)
# --8<-- [end:config_seg_steps]

# %%
# --8<-- [start:config_seg_augs]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    num_epochs=30,
    augmentations=["x_flip", "y_flip", "rotate_90"],  # (1)!
)
# --8<-- [end:config_seg_augs]

# %%
# --8<-- [start:config_seg_channels]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    n_channels_in=3,  # (2)!
    num_epochs=30,
)
# --8<-- [end:config_seg_channels]

# %%
# --8<-- [start:config_seg_prediction]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    prediction_output="probabilities",  # (1)!
)
# --8<-- [end:config_seg_prediction]

# %%
# --8<-- [start:config_seg_val]
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    n_val_patches=15,  # (1)!
)
# --8<-- [end:config_seg_val]

# %%
# --8<-- [start:adv_config_seg_in_memory]
from careamics.config import create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    in_memory=True,  # (1)!
)
# --8<-- [end:adv_config_seg_in_memory]

# %%
# --8<-- [start:adv_config_seg_subchannels]
from careamics.config import create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    channels=[0, 2],  # (1)!
)
# --8<-- [end:adv_config_seg_subchannels]

# %%
# --8<-- [start:adv_config_seg_norm]
from careamics.config import create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    normalization="quantile",  # (1)!
    normalization_params={"lower_quantiles": [0.01], "upper_quantiles": [0.99]},  # (2)!
)
# --8<-- [end:adv_config_seg_norm]

# %%
# --8<-- [start:adv_config_seg_loss]
from careamics.config import create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    loss="dice_ce",  # (1)!
    loss_parameters={
        "class_weights": [1.0, 2.0, 3.0],  # (2)!
        "dice_weight": 0.5,
        "ce_weight": 0.5,
    },
)
# --8<-- [end:adv_config_seg_loss]

# %%
# --8<-- [start:adv_config_seg_filter]
from careamics.config import MaxPatchFilterConfig, create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    patch_filter_config=MaxPatchFilterConfig(
        threshold=0.1,  # (1)!
        filtered_patch_prob=0.2,
    ),
)
# --8<-- [end:adv_config_seg_filter]

# %%
# --8<-- [start:adv_config_seg_model]
from careamics.config import create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    model_params={
        "depth": 3,  # (1)!
        "num_channels_init": 64,
        "use_batch_norm": True,
    },
)
# --8<-- [end:adv_config_seg_model]

# %%
# --8<-- [start:adv_config_seg_num_workers]
from careamics.config import create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    num_workers=4,  # (1)!
)
# --8<-- [end:adv_config_seg_num_workers]

# %%
# --8<-- [start:adv_config_seg_seed]
from careamics.config import create_advanced_seg_config

# create a configuration
config = create_advanced_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    seed=42,
)
# --8<-- [end:adv_config_seg_seed]

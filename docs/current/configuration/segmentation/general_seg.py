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
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,  # (1)!
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
    n_val_patches=15,
)
# --8<-- [end:config_seg_val]

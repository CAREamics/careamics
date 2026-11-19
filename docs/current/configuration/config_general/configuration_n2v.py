#!/usr/bin/env python

# %%
# --8<-- [start:all_configs]
from careamics.config import create_advanced_n2v_config, create_n2v_config

# --8<-- [end:all_configs]

# %%
# --8<-- [start:config_n2v]
from careamics.config import create_n2v_config

config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],  # (1)!
    batch_size=8,
    num_epochs=30,
)
# --8<-- [end:config_n2v]

# %%
# --8<-- [start:config_n2v_steps]
from careamics.config import create_n2v_config

config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    num_steps=500,  # (1)!
)
# --8<-- [end:config_n2v_steps]

# %%
# --8<-- [start:config_n2v_augs]
from careamics.config import create_n2v_config

config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    augmentations=["x_flip", "y_flip", "rotate_90"],  # (1)!
)
# --8<-- [end:config_n2v_augs]

# %%
# --8<-- [start:config_n2v_channels]
from careamics.config import create_n2v_config

config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_channels=3,  # (2)!
)
# --8<-- [end:config_n2v_channels]

# %%
# --8<-- [start:config_n2v_val]
from careamics.config import create_n2v_config

config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_val_patches=15,
)
# --8<-- [end:config_n2v_val]

# %%
# --8<-- [start:config_n2v2]
from careamics.config import create_n2v_config

config = create_n2v_config(
    experiment_name="n2v2_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    use_n2v2=True,
)
# --8<-- [end:config_n2v2]

# %%
# --8<-- [start:adv_config_n2v_params]
from careamics.config import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    roi_size=13,  # (1)!
    masked_pixel_percentage=0.2,  # (2)!
)
# --8<-- [end:adv_config_n2v_params]

# %%
# --8<-- [start:adv_config_structn2v]
from careamics.config import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="struct_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    struct_n2v_axes="horizontal",  # (1)!
    struct_n2v_span=5,  # (2)!
)
# --8<-- [end:adv_config_structn2v]

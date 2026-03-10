#!/usr/bin/env python

############################################
################ Noise2Void ################
# %%
# --8<-- [start:config_n2v]
from careamics.config.ng_factories import create_n2v_config

# create a configuration
config = create_n2v_config(
    experiment_name="n2v_training",  # (1)!
    data_type="array",  # (2)!
    axes="ZYX",  # (3)!
    patch_size=[16, 64, 64],  # (4)!
    batch_size=8,  # (5)!
    num_epochs=30,  # (6)!
)
# --8<-- [end:config_n2v]

# %%
# --8<-- [start:config_n2v_steps]
from careamics.config.ng_factories import create_n2v_config

# create a configuration
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
from careamics.config.ng_factories import create_n2v_config

# create a configuration
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
from careamics.config.ng_factories import create_n2v_config

# create a configuration
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
from careamics.config.ng_factories import create_n2v_config

# create a configuration
config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_val_patches=15,  # (1)!
)
# --8<-- [end:config_n2v_val]


############################################
################# CARE/N2N #################
# %%
# --8<-- [start:config_care]
from careamics.config.ng_factories import create_care_config

# create a configuration
config = create_care_config(
    experiment_name="care_training",  # (1)!
    data_type="array",  # (2)!
    axes="ZYX",  # (3)!
    patch_size=[16, 64, 64],  # (4)!
    batch_size=8,  # (5)!
    num_epochs=30,  # (6)!
)
# --8<-- [end:config_care]

# %%
# --8<-- [start:config_n2n]
from careamics.config.ng_factories import create_n2n_config

# create a configuration
config = create_n2n_config(
    experiment_name="n2n_training",  # (1)!
    data_type="array",  # (2)!
    axes="ZYX",  # (3)!
    patch_size=[16, 64, 64],  # (4)!
    batch_size=8,  # (5)!
    num_epochs=30,  # (6)!
)
# --8<-- [end:config_n2n]

# %%
# --8<-- [start:config_care_steps]
from careamics.config.ng_factories import create_care_config

# create a configuration
config = create_care_config(
    experiment_name="care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    num_steps=500,  # (1)!
)
# --8<-- [end:config_care_steps]

# %%
# --8<-- [start:config_care_augs]
from careamics.config.ng_factories import create_care_config

# create a configuration
config = create_care_config(
    experiment_name="care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    augmentations=["x_flip", "y_flip", "rotate_90"],  # (1)!
)
# --8<-- [end:config_care_augs]

# %%
# --8<-- [start:config_care_channels]
from careamics.config.ng_factories import create_care_config

# create a configuration
config = create_care_config(
    experiment_name="care_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_channels_in=3,  # (2)!
    n_channels_out=2,  # (3)!
)
# --8<-- [end:config_care_channels]

# %%
# --8<-- [start:config_care_val]
from careamics.config.ng_factories import create_care_config

# create a configuration
config = create_care_config(
    experiment_name="care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_val_patches=15,  # (1)!
)
# --8<-- [end:config_care_val]

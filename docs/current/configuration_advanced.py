#!/usr/bin/env python

############################################
################ Noise2Void ################

# %%
# --8<-- [start:adv_config_n2v_in_memory]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="tiff",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    in_memory=True,
)
# --8<-- [end:adv_config_n2v_in_memory]
# %%
# --8<-- [start:adv_config_n2v_subchannels]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    channels=[0, 2],  # (2)!
)
# --8<-- [end:adv_config_n2v_subchannels]
# %%
# --8<-- [start:adv_config_n2v_ind_channels]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_channels=3,  # (2)!
    independent_channels=False,  # (3)!
)
# --8<-- [end:adv_config_n2v_ind_channels]
# %%
# --8<-- [start:adv_config_n2v_norm]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    normalization="quantile",  # (1)!
    normalization_params={"lower_quantiles": [0.01], "upper_quantiles": [0.99]},  # (2)!
)
# --8<-- [end:adv_config_n2v_norm]

# %%
# --8<-- [start:adv_config_n2v_norm_params]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    normalization="mean_std",  # (1)!
    normalization_params={"input_means": [158.95], "input_stds": [10.21]},  # (2)!
)
# --8<-- [end:adv_config_n2v_norm_params]
# %%
# --8<-- [start:adv_config_n2v_norm_ch]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_channels=2,  # (2)!
    normalization="quantile",
    normalization_params={
        "lower_quantiles": [0.01, 0.03],  # (3)!
        "upper_quantiles": [0.99, 0.99],
        "per_channel": True,  # (4)!
    },
)
# --8<-- [end:adv_config_n2v_norm_ch]
# %%
# --8<-- [start:adv_config_n2v_logger]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    logger="wandb",  # (1)!
)
# --8<-- [end:adv_config_n2v_logger]
# %%
# --8<-- [start:adv_config_n2v_num_workers]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    num_workers=4,  # (1)!
)
# --8<-- [end:adv_config_n2v_num_workers]

# %%
# --8<-- [start:adv_config_n2v_seed]
from careamics.config.factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="adv_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    seed=42,
)
# --8<-- [end:adv_config_n2v_seed]

############################################
################# CARE/N2N #################
# %%
# --8<-- [start:adv_config_care_in_memory]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="tiff",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    in_memory=True,
)
# --8<-- [end:adv_config_care_in_memory]
# %%
# --8<-- [start:adv_config_care_subchannels]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    channels=[0, 2],  # (2)!
)
# --8<-- [end:adv_config_care_subchannels]
# %%
# --8<-- [start:adv_config_care_ind_channels]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_channels_in=3,  # (2)!
    independent_channels=False,  # (3)!
)
# --8<-- [end:adv_config_care_ind_channels]
# %%
# --8<-- [start:adv_config_care_norm]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    normalization="quantile",  # (1)!
    normalization_params={"lower_quantiles": [0.01], "upper_quantiles": [0.99]},  # (2)!
)
# --8<-- [end:adv_config_care_norm]

# %%
# --8<-- [start:adv_config_care_norm_params]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    normalization="mean_std",  # (1)!
    normalization_params={
        "input_means": [158.95],  # (2)!
        "input_stds": [10.21],
        "target_means": [121.35],  # (3)!
        "target_stds": [15.78],
    },
)
# --8<-- [end:adv_config_care_norm_params]
# %%
# --8<-- [start:adv_config_care_norm_ch]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="CYX",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    n_channels_in=2,  # (2)!
    normalization="quantile",
    normalization_params={
        "lower_quantiles": [0.01, 0.03],  # (3)!
        "upper_quantiles": [0.99, 0.99],
        "per_channel": True,  # (4)!
    },
)
# --8<-- [end:adv_config_care_norm_ch]

# %%
# --8<-- [start:adv_config_care_logger]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    logger="wandb",  # (1)!
)
# --8<-- [end:adv_config_care_logger]
# %%
# --8<-- [start:adv_config_care_num_workers]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    num_workers=4,  # (1)!
)
# --8<-- [end:adv_config_care_num_workers]

# %%
# --8<-- [start:adv_config_care_seed]
from careamics.config.factories import create_advanced_care_config

# create a configuration
config = create_advanced_care_config(
    experiment_name="adv_care_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    seed=42,
)
# --8<-- [end:adv_config_care_seed]

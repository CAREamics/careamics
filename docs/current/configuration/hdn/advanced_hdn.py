"""Advanced HDN configuration examples for the documentation."""

# %%
# --8<-- [start:adv_config_hdn]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    use_noise_model=False,
    output_channels=1,
    reconstruction_weight=1.0,
    kl_weight=1.0,
    logvar_lowerbound=-5.0,
    optimizer="Adamax",
    optimizer_params={"lr": 3e-4},
    num_epochs=100,
)
# --8<-- [end:adv_config_hdn]

# %%
# --8<-- [start:hdn_noise_model]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_noise_model",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    use_noise_model=True,
    output_channels=1,
)
# --8<-- [end:hdn_noise_model]

# %%
# --8<-- [start:hdn_supervised]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_supervised",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    supervised=True,
    use_noise_model=False,
    output_channels=1,
)
# --8<-- [end:hdn_supervised]

# %%
# --8<-- [start:adv_config_hdn_in_memory]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    in_memory=True,  # (1)!
)
# --8<-- [end:adv_config_hdn_in_memory]

# %%
# --8<-- [start:adv_config_hdn_norm]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    normalization="quantile",  # (1)!
    normalization_params={"lower_quantiles": [0.01], "upper_quantiles": [0.99]},  # (2)!
)
# --8<-- [end:adv_config_hdn_norm]

# %%
# --8<-- [start:adv_config_hdn_norm_params]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    normalization="mean_std",
    normalization_params={"input_means": [158.95], "input_stds": [10.21]},  # (1)!
)
# --8<-- [end:adv_config_hdn_norm_params]

# %%
# --8<-- [start:adv_config_hdn_logger]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    logger="wandb",  # (1)!
)
# --8<-- [end:adv_config_hdn_logger]

# %%
# --8<-- [start:adv_config_hdn_num_workers]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    num_workers=4,  # (1)!
)
# --8<-- [end:adv_config_hdn_num_workers]

# %%
# --8<-- [start:adv_config_hdn_loss]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    reconstruction_weight=1.0,
    kl_weight=1.0,
    logvar_lowerbound=-5.0,
)
# --8<-- [end:adv_config_hdn_loss]

# %%
# --8<-- [start:adv_config_hdn_seed]
from careamics.config import create_advanced_hdn_config

config = create_advanced_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    seed=42,
)
# --8<-- [end:adv_config_hdn_seed]

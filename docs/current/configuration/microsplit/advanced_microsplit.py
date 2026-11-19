"""Advanced MicroSplit configuration examples for the documentation."""

# %%
# --8<-- [start:adv_config_microsplit]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    channels=[0, 2],
    multiscale_count=3,
    padding_mode="reflect",
    normalization="mean_std",
    optimizer="Adamax",
    optimizer_params={"lr": 1e-3},
    num_epochs=100,
)
# --8<-- [end:adv_config_microsplit]

# %%
# --8<-- [start:adv_config_microsplit_in_memory]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    in_memory=True,  # (1)!
)
# --8<-- [end:adv_config_microsplit_in_memory]

# %%
# --8<-- [start:adv_config_microsplit_channels]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    channels=[0, 2],  # (1)!
)
# --8<-- [end:adv_config_microsplit_channels]

# %%
# --8<-- [start:adv_config_microsplit_multiscale]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    multiscale_count=3,  # (1)!
)
# --8<-- [end:adv_config_microsplit_multiscale]

# %%
# --8<-- [start:adv_config_microsplit_padding]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    padding_mode="reflect",  # (1)!
)
# --8<-- [end:adv_config_microsplit_padding]

# %%
# --8<-- [start:adv_config_microsplit_norm]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    normalization="quantile",  # (1)!
    normalization_params={"lower_quantiles": [0.01], "upper_quantiles": [0.99]},  # (2)!
)
# --8<-- [end:adv_config_microsplit_norm]

# %%
# --8<-- [start:adv_config_microsplit_norm_params]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    normalization="mean_std",
    normalization_params={"input_means": [158.95], "input_stds": [10.21]},  # (1)!
)
# --8<-- [end:adv_config_microsplit_norm_params]

# %%
# --8<-- [start:adv_config_microsplit_logger]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    logger="wandb",  # (1)!
)
# --8<-- [end:adv_config_microsplit_logger]

# %%
# --8<-- [start:adv_config_microsplit_num_workers]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    num_workers=4,  # (1)!
)
# --8<-- [end:adv_config_microsplit_num_workers]

# %%
# --8<-- [start:adv_config_microsplit_alpha]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    alpha_ranges=[(0.1, 0.9), (0.2, 0.8)],  # (1)!
)
# --8<-- [end:adv_config_microsplit_alpha]

# %%
# --8<-- [start:adv_config_microsplit_uncorrelated]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    uncorrelated_channel_prob=0.2,  # (1)!
)
# --8<-- [end:adv_config_microsplit_uncorrelated]

# %%
# --8<-- [start:adv_config_microsplit_loss]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    reconstruction_weight=1.0,
    kl_weight=1.0,
    gaussian_likelihood_weight=0.1,
    noise_model_likelihood_weight=0.9,
    predict_logvar=True,
    logvar_lowerbound=-5.0,
)
# --8<-- [end:adv_config_microsplit_loss]

# %%
# --8<-- [start:adv_config_microsplit_seed]
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    seed=42,
)
# --8<-- [end:adv_config_microsplit_seed]

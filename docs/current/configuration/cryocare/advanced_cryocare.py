"""Advanced CryoCARE configuration examples from the dev/cryocare branch."""

# --8<-- [start:adv_config_cryocare]
from careamics.config import QuantileConfig, create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    tilt_axis="Y",
    validation_fraction=0.1,
    n_val_patches=50,
    normalization="mean_std",
    optimizer="Adam",
    optimizer_params={"lr": 4e-4},
    num_epochs=100,
)
# --8<-- [end:adv_config_cryocare]


# %%
# --8<-- [start:cryocare_normalization]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_normalization",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    normalization=QuantileConfig(
        lower_quantiles=[0.01],  # (1)!
        upper_quantiles=[0.99],  # (2)!
    ),
)
# --8<-- [end:cryocare_normalization]

# %%
# --8<-- [start:adv_config_cryocare_logger]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    logger="wandb",  # (1)!
)
# --8<-- [end:adv_config_cryocare_logger]

# %%
# --8<-- [start:adv_config_cryocare_num_workers]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_workers=4,  # (1)!
)
# --8<-- [end:adv_config_cryocare_num_workers]

# %%
# --8<-- [start:adv_config_cryocare_seed]
from careamics.config import create_advanced_cryocare_config

config = create_advanced_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    seed=42,
)
# --8<-- [end:adv_config_cryocare_seed]

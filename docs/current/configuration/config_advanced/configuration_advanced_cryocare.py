"""Advanced CryoCARE configuration examples from the dev/cryocare branch."""

# --8<-- [start:adv_config_cryocare]
from careamics.config import create_advanced_cryocare_config

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

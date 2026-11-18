"""Advanced HDN configuration examples for the documentation."""

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

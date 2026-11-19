"""HDN-specific configuration examples for the documentation."""

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

# --8<-- [start:hdn_supervised]
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

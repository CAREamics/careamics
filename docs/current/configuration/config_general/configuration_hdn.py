"""HDN configuration examples for the documentation."""

# --8<-- [start:all_configs]
from careamics.config import create_advanced_hdn_config, create_hdn_config

# --8<-- [end:all_configs]

# --8<-- [start:config_hdn]
from careamics.config import create_hdn_config

config = create_hdn_config(
    experiment_name="hdn_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=100,
)
# --8<-- [end:config_hdn]

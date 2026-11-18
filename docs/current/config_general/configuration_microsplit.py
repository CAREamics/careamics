"""MicroSplit configuration examples for the documentation."""

# --8<-- [start:all_configs]
from careamics.config import create_advanced_microsplit_config, create_microsplit_config

# --8<-- [end:all_configs]

# --8<-- [start:config_microsplit]
from careamics.config import create_microsplit_config

config = create_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    num_epochs=100,
)
# --8<-- [end:config_microsplit]

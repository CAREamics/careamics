"""CryoCARE configuration examples from the dev/cryocare branch."""

# These snippets target the dev/cryocare branch, where CryoCARE is implemented.

# --8<-- [start:all_configs]
from careamics.config import create_advanced_cryocare_config, create_cryocare_config

# --8<-- [end:all_configs]

# --8<-- [start:config_cryocare]
from careamics.config import create_cryocare_config

config = create_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    tilt_axis="Y",
    num_epochs=100,
)
# --8<-- [end:config_cryocare]

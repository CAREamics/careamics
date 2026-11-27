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
    axes="ZYX",  # (1)!
    patch_size=[64, 64, 64],  # (2)!
    batch_size=8,
    tilt_axis="Y",  # (3)!
    num_epochs=100,
)
# --8<-- [end:config_cryocare]

# %%
# --8<-- [start:config_cryocare_steps]
from careamics.config import create_cryocare_config

config = create_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[64, 64, 64],
    batch_size=8,
    tilt_axis="Y",
    num_epochs=100,
    num_steps=200,  # (1)!
)
# --8<-- [end:config_cryocare_steps]

# %%
# --8<-- [start:cryocare_validation]
from careamics.config import create_cryocare_config

config = create_cryocare_config(
    experiment_name="cryocare_validation",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    tilt_axis="Y",
    validation_fraction=0.2,
)
# --8<-- [end:cryocare_validation]

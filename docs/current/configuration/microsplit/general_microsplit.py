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
    output_channels=3,  # (1)!
    num_epochs=100,
)
# --8<-- [end:config_microsplit]

# %%
# --8<-- [start:config_msplit_steps]
from careamics.config import create_microsplit_config

config = create_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,
    num_epochs=30,
    num_steps=500,  # (1)!
)
# --8<-- [end:config_msplit_steps]

# %%
# --8<-- [start:config_microsplit_augs]
from careamics.config import create_microsplit_config

config = create_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_channels_out=3,
    augmentations=["x_flip"],  # (1)!
)
# --8<-- [end:config_microsplit_augs]

# %%
# --8<-- [start:config_microsplit_val]
from careamics.config import create_microsplit_config

config = create_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_channels_out=3,
    n_val_patches=15,
)
# --8<-- [end:config_microsplit_val]

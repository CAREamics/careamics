#!/usr/bin/env python

# %%
# --8<-- [start:config_n2v2]
from careamics.config.ng_factories import create_n2v_config

# create a configuration
config = create_n2v_config(
    experiment_name="n2v2_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    use_n2v2=True,  # (1)!
)
# --8<-- [end:config_n2v2]

# %%
# --8<-- [start:adv_config_n2v_params]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    roi_size=13,  # (1)!
    masked_pixel_percentage=0.2,  # (2)!
)
# --8<-- [end:adv_config_n2v_params]

# %%
# --8<-- [start:adv_config_structn2v]
from careamics.config.ng_factories import create_advanced_n2v_config

# create a configuration
config = create_advanced_n2v_config(
    experiment_name="struct_n2v_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    struct_n2v_axis="horizontal",  # (1)!
    struct_n2v_span=5,  # (2)!
)
# --8<-- [end:adv_config_structn2v]

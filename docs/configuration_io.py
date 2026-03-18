#!/usr/bin/env python

# %%
# --8<-- [start:save_load]
from careamics.config.ng_factories import create_n2v_config
from careamics.config.utils.configuration_io import (
    save_configuration,
    load_configuration_ng,
)

# create a configuration
config = create_n2v_config(
    experiment_name="n2v",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
)

# save the configuration
config_path = save_configuration(config, "careamics_config.yml")

# load the configuration
loaded_config = load_configuration_ng(config_path)
# --8<-- [end:config]

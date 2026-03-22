#!/usr/bin/env python
# %%
# --8<-- [start:save]
from careamics.config.utils.configuration_io import save_configuration
from careamics.config import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="Config_to_save",
    data_type="tiff",
    axes="ZYX",
    patch_size=(8, 64, 64),
    batch_size=8,
    num_epochs=20,
)
save_configuration(config, "config.yml")
# --8<-- [end:save]

# %%
# --8<-- [start:load]
from careamics.config.utils.configuration_io import load_configuration

config = load_configuration("config.yml")
# --8<-- [end:load]

#!/usr/bin/env python
# %%
from careamics.config import create_n2v_configuration

# %%
create_n2v_configuration(
    experiment_name="N2V_example",
    data_type="arrray",
    axes="YX",
    patch_size=[256, 256],
    batch_size=8,
    num_epochs=30,
)

# %%
create_n2v_configuration(
    experiment_name="N2V_example",
    data_type="array",
    axes="YX",
    patch_size=[256, 256, 512],
    batch_size=8,
    num_epochs=30,
)

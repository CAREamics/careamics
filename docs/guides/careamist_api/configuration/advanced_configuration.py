#!/usr/bin/env python
# %%
# --8<-- [start:data]
from careamics.config import DataConfig

data_config = DataConfig(
    data_type="custom",  # (1)!
    axes="YX",
    patch_size=[128, 128],
    batch_size=8,
    num_epochs=20,
)
# --8<-- [end:data]

#!/usr/bin/env python
import numpy as np
from careamics.config.ng_factories import create_n2v_config, create_care_config

# create a configuration
config_n2v = create_n2v_config(
    experiment_name="n2v",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
)
config_care = create_care_config(
    experiment_name="care",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
)
config = config_n2v

train_data = np.random.randint(0, 255, (512, 512)).astype(np.float32)
train_target = train_data
val_data = train_data
val_target = train_data
pred_data = train_data

# %%
# --8<-- [start:careamist_from_cfg]
from careamics.careamist_v2 import CAREamistV2

careamist = CAREamistV2(config)  # (1)!

# --8<-- [end:careamist_from_cfg]
# %%
# --8<-- [start:careamist_workdir]
from careamics.careamist_v2 import CAREamistV2

careamist = CAREamistV2(
    config,
    work_dir="path/to/work_dir",  # (1)!
)

# --8<-- [end:careamist_workdir]

# %%
# --8<-- [start:train_n2v_no_val]
careamist.train(train_data=train_data)  # (1)!

# --8<-- [end:train_n2v_no_val]

careamist = CAREamistV2(config_care)
# %%
# --8<-- [start:train_care_no_val]
careamist.train(
    train_data=train_data,  # (1)!
    train_data_target=train_target,  # (2)!
)

# --8<-- [end:train_care_no_val]

# %%
# --8<-- [start:train_n2v_val]
careamist.train(
    train_data=train_data,
    val_data=val_data,  # (1)!
)

# --8<-- [end:train_n2v_val]

careamist = CAREamistV2(config_care)
# %%
# --8<-- [start:train_care_val]
careamist.train(
    train_data=train_data,
    train_data_target=train_target,
    val_data=val_data,  # (1)!
    val_data_target=val_target,  # (2)!
)

# --8<-- [end:train_care_val]

#!/usr/bin/env python

# %%
import numpy as np

train_data = np.random.randint(0, 255, (128, 128)).astype(np.float32)
train_target = train_data
val_data = train_data
val_target = train_data
pred_data = train_data

# %%
# --8<-- [start:quick_start_n2v]
from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories import create_n2v_config

# create a configuration
config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
)

# instantiate a careamist
careamist = CAREamistV2(config)

# train the model
careamist.train(train_data=train_data)  # (2)!

# once trained, predict
prediction = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_n2v]

# --8<-- [start:quick_start_care]
from careamics.careamist_v2 import CAREamistV2
from careamics.config.ng_factories import create_care_config

# create a configuration
config = create_care_config(
    experiment_name="care_2D",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
)

# instantiate a careamist
careamist = CAREamistV2(config)

# train the model
careamist.train(
    train_data=train_data,  # (2)!
    train_target=train_target,
    val_data=val_data,
    val_target=val_target,
)

# once trained, predict
prediction = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_care]

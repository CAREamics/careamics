#!/usr/bin/env python
# %%
# --8<-- [start:quick_start_n2v]
import numpy as np

from careamics import CAREamist
from careamics.config import create_n2v_configuration

# create a configuration
config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,  # (1)!
)

# instantiate a careamist
careamist = CAREamist(config)

# train the model
train_data = np.random.randint(0, 255, (256, 256)).astype(np.float32)  # (2)!
careamist.train(train_source=train_data)

# once trained, predict
pred_data = np.random.randint(0, 255, (128, 128)).astype(np.float32)
prediction = careamist.predict(source=pred_data)
# --8<-- [end:quick_start_n2v]

# --8<-- [start:quick_start_care]
import numpy as np

from careamics import CAREamist
from careamics.config import create_care_configuration

# create a configuration
config = create_care_configuration(
    experiment_name="care_2D",
    data_type="array",
    axes="SYX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,  # (1)!
)

# instantiate a careamist
careamist = CAREamist(config)

# train the model
train_data = np.random.randint(0, 255, (5, 256, 256)).astype(np.float32)  # (2)!
train_target = np.random.randint(0, 255, (5, 256, 256)).astype(np.float32)
val_data = np.random.randint(0, 255, (2, 256, 256)).astype(np.float32)
val_target = np.random.randint(0, 255, (2, 256, 256)).astype(np.float32)
careamist.train(
    train_source=train_data,
    train_target=train_target,
    val_source=val_data,
    val_target=val_target,
)

# once trained, predict
pred_data = np.random.randint(0, 255, (128, 128)).astype(np.float32)
prediction = careamist.predict(source=pred_data, axes="YX")
# --8<-- [end:quick_start_care]

# --8<-- [start:quick_start_n2n]
import numpy as np

from careamics import CAREamist
from careamics.config import create_n2n_configuration

# create a configuration
config = create_n2n_configuration(
    experiment_name="n2n_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,  # (1)!
)

# instantiate a careamist
careamist = CAREamist(config)

# train the model
train_data = np.random.randint(0, 255, (256, 256)).astype(np.float32)  # (2)!
train_target = np.random.randint(0, 255, (256, 256)).astype(np.float32)
careamist.train(
    train_source=train_data,
    train_target=train_target,
)

# once trained, predict
pred_data = np.random.randint(0, 255, (128, 128)).astype(np.float32)
prediction = careamist.predict(source=pred_data)
# --8<-- [end:quick_start_care]

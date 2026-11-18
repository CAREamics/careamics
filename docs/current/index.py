#!/usr/bin/env python

# %%
from pathlib import Path

import numpy as np

# cannot be 1 patch because of n_val_patches
train_data = np.random.randint(0, 255, (64 * 4, 64 * 8)).astype(np.float32)
train_target = train_data
val_data = train_data
val_target = train_data
pred_data = train_data
train_volume = np.random.randint(0, 255, (16, 64, 64)).astype(np.float32)
segmentation_target = np.random.randint(0, 2, (16, 64, 64)).astype(np.int64)

# noise model
noise_model_path = Path("dummy_noise_model.npz")
np.savez(
    noise_model_path,
    trained_weight=np.ones((9, 3), dtype=np.float32),
    min_signal=np.array([0.0]),
    max_signal=np.array([255.0]),
    min_sigma=np.array([0.125]),
)
noise_model = [noise_model_path]


# %%
# --8<-- [start:quick_start_n2v]
from careamics.careamist import CAREamist
from careamics.config import create_n2v_config

# create a configuration
config = create_n2v_config(
    experiment_name="n2v_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
)

# instantiate a careamist object
careamist = CAREamist(config)

# train the model
careamist.train(train_data=train_data)  # (2)!

# once trained, predict on data
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_n2v]

# --8<-- [start:quick_start_care]
from careamics.careamist import CAREamist
from careamics.config import create_care_config

# create a configuration
config = create_care_config(
    experiment_name="care_2D",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
)

# instantiate a careamist object
careamist = CAREamist(config)

# train the model
careamist.train(
    train_data=train_data,  # (2)!
    train_data_target=train_target,  # (3)!
    val_data=val_data,
    val_data_target=val_target,
)

# once trained, predict on data
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_care]

# --8<-- [start:quick_start_hdn]
from careamics.careamist import CAREamist
from careamics.config import create_hdn_config

# create a configuration
config = create_hdn_config(
    experiment_name="hdn_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    use_noise_model=True,  # (2)!
)

# instantiate a careamist object
careamist = CAREamist(config)

# train the model
careamist.train(
    train_data=train_data,  # (3)!
    noise_model=noise_model,  # (4)!
)

# once trained, predict on data
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_hdn]

# --8<-- [start:quick_start_microsplit]
from careamics.careamist import CAREamist
from careamics.config import create_microsplit_config

# create a configuration
config = create_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    num_epochs=30,
)

# instantiate a careamist object
careamist = CAREamist(config)

# train the model
careamist.train(train_data=train_data)

# once trained, predict on data
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_microsplit]

# --8<-- [start:quick_start_cryocare]
from careamics.careamist import CAREamist
from careamics.config import create_cryocare_config

config = create_cryocare_config(
    experiment_name="cryocare_training",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    tilt_axis="Y",
    num_epochs=30,
)

careamist = CAREamist(config)
careamist.train(train_data=train_volume)
predictions, sources = careamist.predict(pred_data=train_volume)
# --8<-- [end:quick_start_cryocare]

# --8<-- [start:quick_start_segmentation]
from careamics.careamist import CAREamist
from careamics.config import create_seg_config

config = create_seg_config(
    experiment_name="segmentation_training",
    data_type="array",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    n_classes=2,
    num_epochs=30,
)

careamist = CAREamist(config)
careamist.train(
    train_data=train_volume,
    train_data_target=segmentation_target,
)
predictions, sources = careamist.predict(pred_data=train_volume)
# --8<-- [end:quick_start_segmentation]

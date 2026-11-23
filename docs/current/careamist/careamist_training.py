#!/usr/bin/env python
import numpy as np
from careamics.config import (
    create_n2v_config,
    create_care_config,
    create_hdn_config,
    create_microsplit_config,
    create_seg_config,
    create_cryocare_config,
)

# create a configuration
config_n2v = create_n2v_config(
    experiment_name="n2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
    num_steps=1,
    n_val_patches=2,
)
config_care = create_care_config(
    experiment_name="care",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
    num_steps=1,
    n_val_patches=2,
)
config = config_n2v
config_hdn = create_hdn_config(
    experiment_name="hdn",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
    num_steps=1,
    n_val_patches=2,
)
config_microsplit = create_microsplit_config(
    experiment_name="microsplit",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    num_epochs=1,
    num_steps=1,
)
config_segmentation = create_seg_config(
    experiment_name="segmentation",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    n_classes=2,
    num_epochs=1,
    num_steps=1,
    n_val_patches=2,
)
config_cryocare = create_cryocare_config(
    experiment_name="cryocare",
    data_type="mrc",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=1,
    num_steps=1,
)

train_data = np.random.randint(0, 255, (512, 512)).astype(np.float32)
train_target = train_data
val_data = train_data
val_target = train_data
pred_data = train_data
train_labels = np.random.randint(0, 2, train_data.shape).astype(np.int64)
even_reconstruction = np.random.randint(0, 255, (16, 64, 64)).astype(np.float32)
odd_reconstruction = even_reconstruction.copy()
mask_data = train_data > 50
noise_model = ["noise_models/noise_model_ch0.npz"]

read_func = None  # fake it

# %%
# --8<-- [start:careamist_from_cfg]
from careamics.careamist import CAREamist

careamist = CAREamist(config)  # (1)!

# --8<-- [end:careamist_from_cfg]

# %%
# --8<-- [start:careamist_workdir]
from careamics.careamist import CAREamist

careamist = CAREamist(
    config,
    work_dir="path/to/work_dir",  # (1)!
)

# --8<-- [end:careamist_workdir]

# %%
# --8<-- [start:careamist_pb]
from careamics.careamist import CAREamist

careamist = CAREamist(config, enable_progress_bar=False)
# --8<-- [end:careamist_pb]

from lightning.pytorch.callbacks import Callback


class CustomCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        pass


# %%
# --8<-- [start:callbacks]
from careamics.config import create_advanced_care_config

config = create_advanced_care_config(
    experiment_name="care",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=30,
    checkpoint_params={
        "save_top_k": 5,
        "monitor": "val_loss",
    },
)
config.training_config.early_stopping_params = {  # (1)!
    "monitor": "val_loss",
    "patience": 10,
    "mode": "min",
}

careamist = CAREamist(
    config,  # (2)!
    callbacks=[
        CustomCallback(),  # (3)!
    ],
)
# --8<-- [end:callbacks]

############################################
################ Noise2Void ################
careamist = CAREamist(config_n2v)
# %%
# --8<-- [start:train_n2v_no_val]
careamist.train(train_data=train_data)  # (1)!

# --8<-- [end:train_n2v_no_val]

# --8<-- [start:train_hdn_no_val]
careamist = CAREamist(config_hdn)
careamist.train(train_data=train_data)
# --8<-- [end:train_hdn_no_val]

# --8<-- [start:train_cryocare_no_val]
careamist = CAREamist(config_cryocare)
careamist.train(
    train_data=even_reconstruction,
    train_data_target=odd_reconstruction,
)
# --8<-- [end:train_cryocare_no_val]

# --8<-- [start:train_segmentation_no_val]
careamist = CAREamist(config_segmentation)
careamist.train(
    train_data=train_data,
    train_data_target=train_labels,
)
# --8<-- [end:train_segmentation_no_val]
# %%
# --8<-- [start:train_n2v_val]
careamist.train(
    train_data=train_data,
    val_data=val_data,  # (1)!
)

# --8<-- [end:train_n2v_val]
# %%
# --8<-- [start:train_n2v_mask]
careamist.train(
    train_data=train_data,
    filtering_mask=mask_data,  # (1)!
)

# --8<-- [end:train_n2v_mask]
# %%
# --8<-- [start:train_n2v_custom]
careamist.train(
    train_data=train_data,
    loading=read_func,  # (1)!
)

# --8<-- [end:train_n2v_custom]

############################################
################ CARE ######################

careamist = CAREamist(config_care)
# %%
# --8<-- [start:train_care_no_val]
careamist.train(
    train_data=train_data,  # (1)!
    train_data_target=train_target,  # (2)!
)

# --8<-- [end:train_care_no_val]


careamist = CAREamist(config_care)
# %%
# --8<-- [start:train_care_val]
careamist.train(
    train_data=train_data,
    train_data_target=train_target,
    val_data=val_data,  # (1)!
    val_data_target=val_target,  # (2)!
)

# --8<-- [end:train_care_val]


# --8<-- [start:train_hdn_val]
careamist = CAREamist(config_hdn)
careamist.train(
    train_data=train_data,
    val_data=val_data,  # (1)!
)
# --8<-- [end:train_hdn_val]


# --8<-- [start:train_microsplit_val]
careamist = CAREamist(config_microsplit)
careamist.train(
    train_data=train_data,
    train_data_target=train_data,
    val_data=val_data,  # (1)!
    val_data_target=val_data,  # (2)!
)
# --8<-- [end:train_microsplit_val]


# --8<-- [start:train_cryocare_val]
careamist = CAREamist(config_cryocare)
careamist.train(
    train_data=even_reconstruction,
    train_data_target=odd_reconstruction,  # (1)!
)
# --8<-- [end:train_cryocare_val]


# --8<-- [start:train_segmentation_val]
careamist = CAREamist(config_segmentation)
careamist.train(
    train_data=train_data,
    train_data_target=train_labels,
    val_data=val_data,  # (1)!
    val_data_target=val_target,  # (2)!
)
# --8<-- [end:train_segmentation_val]
# %%
# --8<-- [start:train_care_mask]
careamist.train(
    train_data=train_data,
    train_data_target=train_target,
    filtering_mask=mask_data,  # (1)!
)

# --8<-- [end:train_care_mask]
# %%
# --8<-- [start:train_care_custom]
careamist.train(
    train_data=train_data,
    train_data_target=train_target,
    loading=read_func,  # (1)!
)

# --8<-- [end:train_care_custom]

# --8<-- [start:train_hdn_custom]
careamist = CAREamist(config_hdn)
careamist.train(train_data=train_data, loading=read_func)
# --8<-- [end:train_hdn_custom]

# --8<-- [start:train_cryocare_custom]
careamist = CAREamist(config_cryocare)
careamist.train(
    train_data=even_reconstruction,
    train_data_target=odd_reconstruction,
    loading=read_func,
)
# --8<-- [end:train_cryocare_custom]

# --8<-- [start:train_segmentation_custom]
careamist = CAREamist(config_segmentation)
careamist.train(
    train_data=train_data,
    train_data_target=train_labels,
    loading=read_func,
)
# --8<-- [end:train_segmentation_custom]

# --8<-- [start:train_hdn_mask]
careamist = CAREamist(config_hdn)
careamist.train(train_data=train_data, filtering_mask=mask_data)
# --8<-- [end:train_hdn_mask]

# --8<-- [start:train_cryocare_mask]
careamist = CAREamist(config_cryocare)
careamist.train(
    train_data=even_reconstruction,
    train_data_target=odd_reconstruction,
    filtering_mask=mask_data,
)
# --8<-- [end:train_cryocare_mask]

# --8<-- [start:train_segmentation_mask]
careamist = CAREamist(config_segmentation)
careamist.train(
    train_data=train_data,
    train_data_target=train_labels,
    filtering_mask=mask_data,
)
# --8<-- [end:train_segmentation_mask]

# --8<-- [start:train_hdn_noise_model]
careamist = CAREamist(config_hdn)
careamist.train(
    train_data=train_data,
    noise_model=noise_model,
)
# --8<-- [end:train_hdn_noise_model]

# --8<-- [start:train_microsplit_noise_model]
careamist = CAREamist(config_microsplit)
careamist.train(
    train_data=train_data,
    noise_model=noise_model,
)
# --8<-- [end:train_microsplit_noise_model]

#!/usr/bin/env python
from pathlib import Path
import numpy as np
from careamics.config.factories import create_n2v_config, create_care_config
from careamics.dataset_ng.factory import ReadFuncLoading
from careamics.careamist_v2 import CAREamistV2
import tifffile
import shutil

root = Path(__file__).parent / "temp_data"
root.mkdir(exist_ok=True)

# create a configuration
config_n2v = create_n2v_config(
    experiment_name="n2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
)
config_care = create_care_config(
    experiment_name="care",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
    n_val_patches=2,
)
config = config_n2v

train_data = np.random.randint(0, 255, (512, 512)).astype(np.float32)
train_target = train_data
val_data = train_data
val_target = train_data
pred_data = train_data[:256, :256]
mask_data = train_data > 50

read_func = None  # fake it

careamist = CAREamistV2(config)
careamist.train(train_data=train_data)

# %%
# --8<-- [start:pred]
predictions = careamist.predict(
    pred_data=pred_data,
)
# --8<-- [end:pred]


# %%
# --8<-- [start:pred_tiled]
predictions = careamist.predict(
    pred_data=pred_data,
    tile_size=[128, 128],  # (1)!
    tile_overlap=[48, 48],  # (2)!
)
# --8<-- [end:pred_tiled]

# %%
# --8<-- [start:pred_dataloader]
predictions = careamist.predict(
    pred_data=pred_data,
    batch_size=4,
    num_workers=0,
)
# --8<-- [end:pred_dataloader]

pred_data_path = root / "pred.tiff"
tifffile.imwrite(pred_data_path, np.stack([pred_data, pred_data, pred_data]))
assert pred_data_path.exists()

# %%
# --8<-- [start:pred_data_params]
predictions = careamist.predict(
    pred_data=pred_data_path,  # (1)!
    axes="SYX",
    data_type="tiff",
    in_memory=False,
)
# --8<-- [end:pred_data_params]


# delete root
shutil.rmtree(root)

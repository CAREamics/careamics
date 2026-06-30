#!/usr/bin/env python
from pathlib import Path
import numpy as np
from careamics.config.factories import create_n2v_config
from careamics.careamist import CAREamist
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
    num_steps=1,
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

careamist = CAREamist(config, work_dir=root)
careamist.train(train_data=train_data)

# %%
# --8<-- [start:pred]
pred_arrays, sources = careamist.predict(
    pred_data=pred_data,
)
# --8<-- [end:pred]

# %%
# --8<-- [start:pred_empty_source]
pred_arrays, _ = careamist.predict(
    pred_data=pred_data,
)
# --8<-- [end:pred_empty_source]

# %%
# --8<-- [start:pred_checkpoint]
pred_arrays, sources = careamist.predict(
    pred_data=pred_data,
    checkpoint="last",  # or "best"
)
# --8<-- [end:pred_checkpoint]

# %%
# --8<-- [start:pred_tiled]
pred_arrays, sources = careamist.predict(
    pred_data=pred_data,
    tile_size=(128, 128),  # (1)!
    tile_overlap=(48, 48),  # (2)!
)
# --8<-- [end:pred_tiled]

# %%
# --8<-- [start:pred_dataloader]
pred_arrays, sources = careamist.predict(
    pred_data=pred_data,
    batch_size=4,
    num_workers=0,
)
# --8<-- [end:pred_dataloader]

# save a tiff to be used as input path
pred_data_path = root / "my_predictions" / "pred.tiff"
pred_data_path.parent.mkdir(parents=True, exist_ok=True)
tifffile.imwrite(pred_data_path, np.stack([pred_data, pred_data, pred_data]))
assert pred_data_path.exists()

# %%
# --8<-- [start:pred_data_params]
pred_arrays, sources = careamist.predict(
    pred_data=pred_data_path,  # (1)!
    axes="SYX",
    data_type="tiff",
    in_memory=False,
)
# --8<-- [end:pred_data_params]

# go back to using an input array
pred_data = train_data[:256, :256]

# %%
# --8<-- [start:pred_to_disk]
careamist.predict_to_disk(
    pred_data=pred_data,
    prediction_dir=root / "my_predictions",  # (1)!
)
# --8<-- [end:pred_to_disk]

# %%
# --8<-- [start:pred_write_type]
careamist.predict_to_disk(
    pred_data=pred_data,
    write_type="tiff",  # (1)!
)
# --8<-- [end:pred_write_type]

# %%
# --8<-- [start:pred_write_func]


def save_to_np(file_path: Path, img: np.ndarray, *args, **kwargs) -> None:  # (1)!
    """Custom write function to save predictions to disk as .npy files."""
    np.save(file_path, img, *args, **kwargs)


careamist.predict_to_disk(
    pred_data=pred_data,
    write_type="custom",  # (2)!
    write_extension=".npy",  # (3)!
    write_func=save_to_np,  # (4)!
    write_func_kwargs={"allow_pickle": False},  # (5)!
)
# --8<-- [end:pred_write_func]

# delete root
shutil.rmtree(root)

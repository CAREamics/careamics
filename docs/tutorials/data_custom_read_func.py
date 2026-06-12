# %%
# --8<-- [start:data-config]

from pathlib import Path
import numpy as np

from careamics import CAREamist, ReadFuncLoading
from careamics.config import create_n2v_config

# --- create toy data
DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

n_files = 5
image_shape = (512, 512)
file_paths: list[Path] = []
for i in range(5):
    image = np.random.rand(*image_shape)
    file_path = DATA_PATH / f"image_{i}.npy"
    np.save(file_path, image)
    file_paths.append(file_path)
# 2D image files in a directory, each with the shape (512, 512)
# data/
# ├── image_0.npy
# ├── image_1.npy
# ├── image_2.npy
# ├── image_3.npy
# └── image_4.npy

# --- configuration
config = create_n2v_config(
    "loading-custom",
    data_type="custom",  # (1)!
    axes="YX",  # (2)!
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=10,
)
# --8<-- [end:data-config]


# --8<-- [start:training]
# Wrapping numpy.load
# The call signature should match the protocol `ReadFunc`
def read_numpy(file_path: Path) -> np.ndarray:
    return np.load(file_path)


careamist = CAREamist(config)
careamist.train(
    train_data=DATA_PATH,  # (1)!
    loading=ReadFuncLoading(  # (2)!
        read_numpy,  # (3)!
        extension_filter="*.npy",  # (4)!
    ),
)
# --8<-- [end:training]

# %%
# --8<-- [start:prediction]
predictions, sources = careamist.predict(  # (1)!
    pred_data=DATA_PATH,
    loading=ReadFuncLoading(
        read_numpy,
        extension_filter="*.npy",
    ),
)
# inspect the sources of the predictions
sources
# --8<-- [start:prediction]

# --8<-- [start:download-data]
from pathlib import Path

import matplotlib.pyplot as plt
import pooch
import tifffile

from careamics.dataset.patch_filter import (
    MaxPatchFilter,
    MeanStdPatchFilter,
    ShannonPatchFilter,
)

# --- download the data
# folder in which to save all the data
root = Path("hagen")

# download the data using pooch
data_root = root / "data"
dataset_url = "https://zenodo.org/records/10925855/files/noisy.tiff?download=1"

file = pooch.retrieve(
    url=dataset_url,
    known_hash="ff12ee5566f443d58976757c037ecef8bf53a00794fa822fe7bcd0dd776a9c0f",
    path=data_root,
)

# Shape: (79, 1024, 1024), axes: SYX
img = tifffile.imread(file)
# --8<-- [end:download-data]


# --8<-- [start:filter-maps]
sample_idx = 4

max_filter_map = MaxPatchFilter.filter_map(img[sample_idx], (64, 64))
MaxPatchFilter.plot_filter_map(img[sample_idx], max_filter_map)

shannon_filter_map = ShannonPatchFilter.filter_map(img[sample_idx], (64, 64))
ShannonPatchFilter.plot_filter_map(img[sample_idx], shannon_filter_map)

meanstd_filter_map = MeanStdPatchFilter.filter_map(img[sample_idx], (64, 64))
MeanStdPatchFilter.plot_filter_map(img[sample_idx], meanstd_filter_map)
# --8<-- [end:filter-maps]


# --8<-- [start:mask]
plt.figure(constrained_layout=True)
plt.imshow(ShannonPatchFilter.apply_filter(shannon_filter_map, threshold=7.5))
plt.title("Filter mask")
# --8<-- [end:mask]


# --8<-- [start:config]
from careamics import CAREamist
from careamics.config import ShannonPatchFilterConfig, create_advanced_n2v_config

config = create_advanced_n2v_config(
    "hagen-shannon-filtering",
    data_type="array",
    axes="SYX",
    patch_size=(64, 64),
    batch_size=64,
    num_epochs=10,
    patch_filter_config=ShannonPatchFilterConfig(threshold=7.5),  # (1)!
)

careamist = CAREamist(config=config)
careamist.train(train_data=img)
# --8<-- [start:config]

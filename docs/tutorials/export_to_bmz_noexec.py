#!/usr/bin/env python
# NOTE: this script is intentionally excluded from `test_guides.py` (see the
# `exclude` list there). Exporting to the BioImage Model Zoo runs a full model
# validation through `bioimageio.core`, which is too heavy to run on every test
# pass. The snippets below are still written to be runnable so they stay in sync
# with the real API.
from pathlib import Path

import numpy as np

from careamics.careamist import CAREamist
from careamics.config.factories import create_n2v_config
from careamics.model_io.bmz_io import load_from_bmz

root = Path(__file__).parent / "temp_data"
root.mkdir(exist_ok=True)

config = create_n2v_config(
    experiment_name="bmz_export",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
    num_steps=1,
    n_val_patches=2,
)

train_data = np.random.randint(0, 255, (512, 512)).astype(np.float32)
input_array = train_data[:64, :64]

careamist = CAREamist(config, work_dir=root)
careamist.train(train_data=train_data)

path_to_archive = root / "my_model.zip"


# %%
# --8<-- [start:export_min]
careamist.export_to_bmz(
    path_to_archive="my_model.zip",
    friendly_model_name="my_n2v_model",
    input_array=input_array,  # (1)!
    authors=[{"name": "Jane Doe"}],
    general_description="A Noise2Void model trained with CAREamics.",
    data_description="Fluorescence microscopy images of cells.",
)
# --8<-- [end:export_min]

# %%
# --8<-- [start:authors]
authors = [
    {"name": "Jane Doe", "affiliation": "My Institute", "github_user": "janedoe"},
    {"name": "John Smith", "affiliation": "My Institute"},
]
# --8<-- [end:authors]

# generate a cover image on disk so the example below has a real file to point to
import matplotlib.pyplot as plt

plt.imsave(root / "cover.png", input_array, cmap="gray")

# %%
# --8<-- [start:export_full]
careamist.export_to_bmz(
    path_to_archive=path_to_archive,
    friendly_model_name="my_n2v_model",
    input_array=input_array,
    authors=authors,
    general_description="A Noise2Void model trained with CAREamics.",
    data_description="Fluorescence microscopy images of cells.",
    covers=[root / "cover.png"],  # (1)!
    channel_names=["nucleus"],  # (2)!
)
# --8<-- [end:export_full]

# %%
# --8<-- [start:load]
config, model = load_from_bmz(path_to_archive)
# --8<-- [end:load]

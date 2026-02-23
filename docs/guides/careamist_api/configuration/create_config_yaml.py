#!/usr/bin/env python
# This script is used to create the config.yaml used in the docs
from pathlib import Path

from careamics import save_configuration
from careamics.config import create_n2v_configuration

config_path = Path(__file__).parent / "config.yml"
config = create_n2v_configuration(
    experiment_name="N2V 3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=(8, 64, 64),
    batch_size=8,
    num_epochs=20,
)
save_configuration(config, config_path)

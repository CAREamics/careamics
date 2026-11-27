"""Noise-model examples for the documentation."""

from functools import partial
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
train_signal = rng.uniform(0, 255, size=(10, 64, 64)).astype(np.float32)
train_observation = np.clip(
    train_signal + rng.normal(0, 10, size=train_signal.shape), 0, 255
).astype(np.float32)
val_signal = rng.uniform(0, 255, size=(4, 64, 64)).astype(np.float32)
val_observation = np.clip(
    val_signal + rng.normal(0, 10, size=val_signal.shape), 0, 255
).astype(np.float32)

# --8<-- [start:noise_model_train]
from careamics.noise_model import train_noise_model

# --8<-- [end:noise_model_train]

train_noise_model = partial(
    train_noise_model,
    n_gaussian=1,
    n_epochs=2,
)

# --8<-- [start:noise_model_train_2]
noise_model = train_noise_model(
    signal=train_signal,
    observation=train_observation,
    axes="SYX",
    save_dir="noise_models",
)
# --8<-- [end:noise_model_train_2]

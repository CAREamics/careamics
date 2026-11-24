"""Noise-model examples for the documentation."""

import numpy as np

rng = np.random.default_rng(42)
signal = rng.uniform(0, 255, size=(10, 64, 64)).astype(np.float32)
observation = signal + rng.normal(0, 10, size=signal.shape).astype(np.float32)

# --8<-- [start:noise_model_train]
from careamics.noise_model import train_noise_model

noise_model = train_noise_model(
    signal=signal,
    observation=observation,
    axes="SYX",
    n_gaussian=1,
    n_coeff=2,
    n_epochs=10,
    save_dir="noise_models",
)
# --8<-- [end:noise_model_train]

# --8<-- [start:noise_model_use_hdn]
from pathlib import Path

from careamics.careamist import CAREamist
from careamics.config import create_hdn_config

config = create_hdn_config(
    experiment_name="hdn_with_noise_model",
    data_type="array",
    axes="SYX",
    patch_size=[64, 64],
    batch_size=8,
    use_noise_model=True,
)

careamist = CAREamist(config)
noise_model_paths = sorted(Path("noise_models").glob("*.npz"))
careamist.train(train_data=observation, noise_model=noise_model_paths)
# --8<-- [end:noise_model_use_hdn]

# --8<-- [start:noise_model_use_microsplit]
from pathlib import Path

from careamics.careamist import CAREamist
from careamics.config import create_advanced_microsplit_config

config = create_advanced_microsplit_config(
    experiment_name="microsplit_with_noise_model",
    data_type="array",
    axes="SYX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=1,
    noise_model_likelihood_weight=1.0,
)

careamist = CAREamist(config)
noise_model_paths = sorted(Path("noise_models").glob("*.npz"))
careamist.train(train_data=observation, noise_model=noise_model_paths)
# --8<-- [end:noise_model_use_microsplit]

#!/usr/bin/env python

from pathlib import Path

import numpy as np

train_data = np.random.randint(0, 255, (64 * 4, 64 * 8)).astype(np.float32)
pred_data = train_data

noise_model = []
for channel in range(3):
    noise_model_path = Path(f"dummy_noise_model_ch{channel}.npz")
    np.savez(
        noise_model_path,
        trained_weight=np.ones((9, 3), dtype=np.float32),
        min_signal=np.array([0.0]),
        max_signal=np.array([255.0]),
        min_sigma=np.array([0.125]),
    )
    noise_model.append(noise_model_path)


# --8<-- [start:quick_start_microsplit_config]
from careamics.config import create_microsplit_config

config = create_microsplit_config(
    experiment_name="microsplit_training",
    data_type="array",  # (1)!
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    output_channels=3,  # (2)!
    use_noise_model=True,  # (3)!
    num_epochs=2,
    num_steps=5,
)
# --8<-- [end:quick_start_microsplit_config]

# --8<-- [start:quick_start_microsplit_train]
from careamics.careamist import CAREamist

careamist = CAREamist(config)
careamist.train(
    train_data=train_data,  # (4)!
    noise_model=noise_model,  # (5)!
)
predictions, sources = careamist.predict(pred_data=pred_data)
# --8<-- [end:quick_start_microsplit_train]

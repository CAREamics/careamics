from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.careamist import CAREamist
from careamics.config.factories import create_advanced_care_config

# skip if fail imports
pylib = pytest.importorskip("pylibCZIrw")

from pylibCZIrw import czi as pyczi  # noqa: E402


# TODO slightly different version that the function in test_czi_image_stack,
#   they should be consolidated.
def create_test_czi(file_path: Path, data: NDArray, axes: str):

    if axes == "SCYX":
        # add a singleton
        data = np.expand_dims(data, axis=2)
        assert len(data.shape) == 5

    shape = data.shape
    with pyczi.create_czi(str(file_path)) as czi:
        xoffs = 0
        for s in range(shape[0]):
            for c in range(shape[1]):
                for k in range(shape[2]):
                    k_key = "T" if "T" in axes else "Z"

                    czi.write(
                        data[s, c, k],
                        plane={"C": c, k_key: k},
                        location=(xoffs, 0),
                        scene=s,
                    )
            xoffs += data.shape[-1] + 20


def test_t_as_z_care(tmp_path):
    """Test e2e training and prediction with CZI."""
    axes = "SCTYX"
    shape = (2, 2, 8, 32, 32)

    # create data
    rng = np.random.default_rng(42)
    train_data = rng.integers(0, 255, shape).astype(np.float32)
    train_target = rng.integers(0, 255, shape).astype(np.float32)

    # create czi files
    train_path = tmp_path / "train" / "train.czi"
    create_test_czi(train_path, train_data, axes)

    target_path = tmp_path / "target" / "train.czi"
    create_test_czi(target_path, train_target, axes)

    # create a configuration
    config = create_advanced_care_config(
        experiment_name="e2e_care_target_axes",
        data_type="czi",
        axes=axes,
        target_axes=axes,
        patch_size=[4, 8, 8],
        batch_size=2,
        num_epochs=1,
        num_steps=2,
        n_channels_in=2,
    )

    # instantiate a careamist
    careamist = CAREamist(config)

    # train the model
    careamist.train(train_data=train_path, train_data_target=target_path)

    # predict
    predictions, _ = careamist.predict(pred_data=train_path)

    assert predictions[0].shape == (1,) + shape[1:]

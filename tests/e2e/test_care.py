import numpy as np

from careamics.careamist import CAREamist
from careamics.config.factories import create_advanced_care_config


def test_target_axes():
    """Test e2e training and prediction with different target axes."""
    axes = "SYXZC"
    in_shape = (2, 32, 32, 8, 3)

    target_axes = "SZYX"
    out_shape = (2, 8, 32, 32)

    # create data
    rng = np.random.default_rng(42)
    train_data = [rng.integers(0, 255, in_shape).astype(np.float32) for _ in range(4)]
    train_target = [
        rng.integers(0, 255, out_shape).astype(np.float32) for _ in range(4)
    ]

    # create a configuration
    config = create_advanced_care_config(
        experiment_name="e2e_care_target_axes",
        data_type="array",
        axes=axes,
        target_axes=target_axes,
        patch_size=[4, 8, 8],
        # batch size, num_epochs and num_steps reduced for the sake of example
        batch_size=2,
        num_epochs=1,
        num_steps=2,
        channels=[0, 2],
    )

    # instantiate a careamist
    careamist = CAREamist(config)

    # train the model
    careamist.train(train_data=train_data, train_data_target=train_target)

    # predict
    predictions, _ = careamist.predict(pred_data=train_data)

    assert predictions[0].shape == out_shape

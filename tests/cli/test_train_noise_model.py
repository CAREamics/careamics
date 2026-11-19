import numpy as np

from careamics.cli.main import main


def test_train_noise_model(tmp_path):
    rng = np.random.default_rng(42)
    signal = rng.uniform(0, 255, (2, 16, 16)).astype(np.float32)
    observation = signal + rng.normal(0, 10, signal.shape).astype(np.float32)
    signal_path = tmp_path / "signal.npy"
    observation_path = tmp_path / "observation.npy"
    output_path = tmp_path / "noise_models"
    np.save(signal_path, signal)
    np.save(observation_path, observation)

    main(
        [
            "train-noise-model",
            "--output_path",
            str(output_path),
            "--signal",
            str(signal_path),
            "--observation",
            str(observation_path),
            "--axes",
            "SYX",
            "--n_epochs",
            "1",
        ]
    )

    assert (output_path / "noise_model_ch0.npz").is_file()

from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr

from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.dataset.factory import ReadFuncLoading
from careamics.models.lvae.noise_model_data_loader import (
    create_noise_model_data_loader,
)
from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


def test_small_array_loader_preserves_number_of_epochs() -> None:
    signal = np.arange(20, dtype=np.float32).reshape(4, 5)
    observation = signal + 1

    batches = list(
        create_noise_model_data_loader(
            signal,
            observation,
            axes="YX",
            data_type="array",
            batch_size=8,
            n_epochs=4,
        )
    )

    assert len(batches) == 4
    assert all(batch.shape == (8, 2) for batch in batches)
    assert all(np.allclose(batch[:, 1], batch[:, 0] + 1) for batch in batches)


def test_large_array_loader_limits_pixels_and_balances_deciles() -> None:
    signal = np.arange(1000, dtype=np.float32).reshape(20, 50)
    observation = signal + 1

    batches = list(
        create_noise_model_data_loader(
            signal,
            observation,
            axes="YX",
            data_type="array",
            batch_size=10,
            n_epochs=3,
        )
    )

    assert len(batches) == 3
    assert sum(len(batch) for batch in batches) == 30
    assert all(np.allclose(batch[:, 1], batch[:, 0] + 1) for batch in batches)


def test_array_sequence_accepts_different_shapes() -> None:
    signals = [
        np.arange(20, dtype=np.float32).reshape(4, 5),
        np.arange(21, dtype=np.float32).reshape(3, 7),
    ]
    observations = [signal + 2 for signal in signals]

    batches = list(
        create_noise_model_data_loader(
            signals,
            observations,
            axes="YX",
            data_type="array",
            batch_size=32,
            n_epochs=2,
        )
    )

    assert len(batches) == 2
    assert all(np.allclose(batch[:, 1], batch[:, 0] + 2) for batch in batches)


def test_tiff_loader_accepts_differently_shaped_file_pairs(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signal"
    observation_dir = tmp_path / "observation"
    signal_dir.mkdir()
    observation_dir.mkdir()
    for name, shape in (("a.tiff", (4, 5)), ("b.tiff", (3, 7))):
        signal = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        tifffile.imwrite(signal_dir / name, signal)
        tifffile.imwrite(observation_dir / name, signal + 2)

    batches = list(
        create_noise_model_data_loader(
            signal_dir,
            observation_dir,
            axes="YX",
            data_type="tiff",
            batch_size=32,
            n_epochs=2,
        )
    )

    assert len(batches) == 2
    assert all(np.allclose(batch[:, 1], batch[:, 0] + 2) for batch in batches)


def test_large_tiff_loader_uses_unpadded_edge_patches(tmp_path: Path) -> None:
    signal = np.arange(7 * 13, dtype=np.float32).reshape(7, 13)
    signal_path = tmp_path / "signal.tiff"
    observation_path = tmp_path / "observation.tiff"
    tifffile.imwrite(signal_path, signal)
    tifffile.imwrite(observation_path, signal + 3)

    batches = list(
        create_noise_model_data_loader(
            signal_path,
            observation_path,
            axes="YX",
            data_type="tiff",
            batch_size=5,
            n_epochs=2,
        )
    )

    assert len(batches) == 2
    assert sum(len(batch) for batch in batches) == 10
    assert all(np.allclose(batch[:, 1], batch[:, 0] + 3) for batch in batches)
    assert all(np.all(batch[:, 0] >= signal.min()) for batch in batches)


def test_custom_read_func_loading(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signal"
    observation_dir = tmp_path / "observation"
    signal_dir.mkdir()
    observation_dir.mkdir()
    signal = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.save(signal_dir / "image.npy", signal)
    np.save(observation_dir / "image.npy", signal + 4)
    loading = ReadFuncLoading(
        read_source_func=np.load,
        extension_filter="*.npy",
    )

    batches = list(
        create_noise_model_data_loader(
            signal_dir,
            observation_dir,
            axes="YX",
            data_type="custom",
            loading=loading,
            batch_size=32,
            n_epochs=1,
        )
    )

    assert len(batches) == 1
    assert np.allclose(batches[0][:, 1], batches[0][:, 0] + 4)


def test_zarr_loader_selects_channel(tmp_path: Path) -> None:
    signal = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    signal_path = tmp_path / "signal.zarr"
    observation_path = tmp_path / "observation.zarr"
    signal_group = zarr.create_group(signal_path)
    observation_group = zarr.create_group(observation_path)
    signal_group.create_array("image", data=signal, chunks=(1, 4, 4))
    observation_group.create_array("image", data=signal + 5, chunks=(1, 4, 4))

    batches = list(
        create_noise_model_data_loader(
            signal_path,
            observation_path,
            axes="CYX",
            data_type="zarr",
            channel=1,
            batch_size=16,
            n_epochs=2,
        )
    )

    assert len(batches) == 2
    assert all(np.allclose(batch[:, 1], batch[:, 0] + 5) for batch in batches)
    assert all(np.all(batch[:, 0] >= signal[1].min()) for batch in batches)


def test_noise_model_fit_accepts_tiff_paths(tmp_path: Path) -> None:
    signal = np.arange(64, dtype=np.float32).reshape(8, 8)
    signal_path = tmp_path / "signal.tiff"
    observation_path = tmp_path / "observation.tiff"
    tifffile.imwrite(signal_path, signal)
    tifffile.imwrite(observation_path, signal + 1)
    model = GaussianMixtureNoiseModel(
        GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            n_gaussian=1,
            min_sigma=1,
            min_signal=float(signal.min()),
            max_signal=float(signal.max()),
        )
    )
    initial_weight = model.weight.detach().clone()

    losses = model.fit(
        signal_path,
        observation_path,
        axes="YX",
        data_type="tiff",
        batch_size=16,
        n_epochs=2,
    )

    assert len(losses) == 2
    assert not torch.equal(model.weight, initial_weight)

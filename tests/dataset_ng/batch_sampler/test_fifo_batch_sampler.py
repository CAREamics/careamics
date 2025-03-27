from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from careamics.config import DataConfig, InferenceConfig
from careamics.dataset_ng.batch_sampler import FifoBatchSampler, FifoImageStackManager
from careamics.dataset_ng.dataset import Mode
from careamics.dataset_ng.dataset.factory import create_lazy_tiff_dataset
from careamics.dataset_ng.patch_extractor.image_stack import ManagedLazyImageStack

LazyCallback = Callable[["ManagedLazyImageStack"], None]


def _create_test_data(save_path: Path) -> tuple[list[Path], str]:
    """Create tiff files"""
    axes = "SCYX"
    data_shapes = [
        (2, 1, 32, 32),
        (1, 1, 19, 37),
        (3, 1, 14, 9),
        (2, 1, 32, 32),
        (1, 1, 19, 37),
        (3, 1, 14, 9),
    ]
    paths: list[Path] = []
    for i, shape in enumerate(data_shapes):
        path = Path(save_path / f"image_{i}.tiff")
        data = np.arange(np.prod(shape)).reshape(shape)
        tifffile.imwrite(path, data)
        paths.append(path)
    return paths, axes


def _wrap_callbacks(
    image_stack: ManagedLazyImageStack,
    on_load_mock: Mock,
    on_close_mock: Mock,
):
    """
    Wrap the ManagedLazyImageStack callbacks so we can know if they have been called.
    """
    original_on_load = image_stack._on_load
    original_on_close = image_stack._on_close

    # wrap the original callbacks so that mocks are called first and then the callback
    def wrapped_on_load(img_s: ManagedLazyImageStack):
        on_load_mock()  # later we can test on_load_mock.assert_called_once
        if original_on_load is not None:
            original_on_load(img_s)

    def wrapped_on_close(img_s: ManagedLazyImageStack):
        on_close_mock()  # later we can test on_close_mock.assert_called_once
        if original_on_close is not None:
            original_on_close(img_s)

    # set the callbacks
    image_stack.set_callbacks(wrapped_on_load, wrapped_on_close)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("shuffle", [True, False])
# TODO add Mode.PREDICTING when #444 has been merged
@pytest.mark.parametrize("mode", [Mode.TRAINING, Mode.VALIDATING])
@pytest.mark.parametrize("max_files_loaded", [1, 2, 3])
def test_fifo_batch_sampler(
    tmp_path,
    minimum_data,
    max_files_loaded: int,
    mode: Mode,
    shuffle: bool,
    batch_size: int,
):
    """"""
    paths, axes = _create_test_data(tmp_path)

    if mode != Mode.PREDICTING:
        data_config = DataConfig(**minimum_data)
        data_config.axes = axes
        data_config.patch_size = [8, 8]
    else:
        del minimum_data["patch_size"]
        minimum_data["tile_size"] = [8, 8]
        minimum_data["overlaps"] = [2, 2]
        minimum_data["image_means"] = [0]
        minimum_data["image_stds"] = [1]
        data_config = InferenceConfig(**minimum_data)

    # create dataset and batch sampler
    dataset = create_lazy_tiff_dataset(
        config=data_config, mode=mode, inputs=paths, targets=None
    )
    batch_sampler = FifoBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        max_files_loaded=max_files_loaded,
        drop_last=True,  # TODO test false
        random_seed=42,
    )

    # wrap the callbacks so we can test if they have been called
    on_load_mocks = [Mock() for _ in paths]
    on_close_mocks = [Mock() for _ in paths]
    for i, image_stack in enumerate(dataset.input_extractor.image_stacks):
        _wrap_callbacks(image_stack, on_load_mocks[i], on_close_mocks[i])

    expected_n_batches = len(dataset) // batch_size  # drop last is True so floor divide
    batch_count = 0
    all_indices = np.array([])
    for batch_indices in batch_sampler:
        all_indices = np.concatenate([all_indices, batch_indices])
        for index in batch_indices:
            _ = dataset[index]  # so that images will be loaded

        # because drop last is True, should always be batch size
        assert len(batch_indices) == batch_size

        # test never more than specified n files open
        assert batch_sampler.fifo_manager.n_currently_loaded <= max_files_loaded

        batch_count += 1
    assert batch_count == expected_n_batches
    # make sure there are no repeated indices
    assert len(np.unique(all_indices)) == len(all_indices)

    # make sure load and closed was called only once for each image
    # we can do this because of the _wrap_callbacks function
    for i in range(len(paths)):
        on_load_mocks[i].assert_called_once()
        on_close_mocks[i].assert_called_once()


@pytest.mark.parametrize("max_files_loaded", [1, 2, 3])
def test_fifo_image_stack_manager(tmp_path, max_files_loaded: int):
    paths, axes = _create_test_data(tmp_path)

    manager = FifoImageStackManager(max_files_loaded)

    image_stacks: list[ManagedLazyImageStack] = []
    on_load_mocks = [Mock() for _ in paths]
    on_close_mocks = [Mock() for _ in paths]
    for i, path in enumerate(paths):

        image_stack = ManagedLazyImageStack.from_tiff(path, axes)
        manager.register_image_stack(image_stack)
        # wrap the callbacks so we can know if they have been called
        _wrap_callbacks(image_stack, on_load_mocks[i], on_close_mocks[i])

        image_stacks.append(image_stack)

    # test never more than specified n files open
    for image_stack in image_stacks:
        image_stack.load()
        assert manager.n_currently_loaded <= max_files_loaded
    manager.close_all()  # close remaining open files

    # make sure load and closed was called only once for each image
    for i in range(len(image_stacks)):
        on_load_mocks[i].assert_called_once()
        on_close_mocks[i].assert_called_once()

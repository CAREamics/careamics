from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from careamics.dataset_ng.batch_sampler import FifoImageStackManager
from careamics.dataset_ng.patch_extractor.image_stack import ManagedLazyImageStack


@pytest.mark.parametrize("max_files_loaded", [1, 2, 3])
def test_fifo_image_stack_manager(tmp_path, max_files_loaded: int):
    manager = FifoImageStackManager(max_files_loaded)

    axes = "SCYX"
    data_shapes = [
        (2, 1, 32, 32),
        (1, 1, 19, 37),
        (3, 1, 14, 9),
        (2, 1, 32, 32),
        (1, 1, 19, 37),
        (3, 1, 14, 9),
    ]
    # write files
    image_stacks: list[ManagedLazyImageStack] = []
    on_load_mocks = [Mock() for _ in data_shapes]
    on_close_mocks = [Mock() for _ in data_shapes]
    for i, shape in enumerate(data_shapes):
        path = Path(tmp_path / f"image_{i}.tiff")
        data = np.arange(np.prod(shape)).reshape(shape)
        tifffile.imwrite(path, data)

        def on_load_closure(index: int):
            def on_load(img):
                on_load_mocks[index]()
                manager.notify_load(img)

            return on_load

        def on_close_closure(index: int):
            def on_close(img):
                on_close_mocks[index]()
                manager.notify_close(img)

            return on_close

        image_stack = ManagedLazyImageStack.from_tiff(path, axes)
        manager.register_image_stack(image_stack)
        # have to do this to override callbacks set when calling register_image_stack
        image_stack.set_callbacks(on_load_closure(i), on_close_closure(i))

        image_stacks.append(image_stack)

    # test never more than specified n files open
    for image_stack in image_stacks:
        image_stack.load()
        assert manager.currently_loaded <= max_files_loaded
    manager.close_all()  # close remaining open files

    # make sure load and closed was called only once for each image
    for i in range(len(image_stacks)):
        on_load_mocks[i].assert_called_once()
        on_close_mocks[i].assert_called_once()

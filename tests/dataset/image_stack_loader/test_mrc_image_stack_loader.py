from pathlib import Path

import mrcfile
import numpy as np

from careamics.dataset.image_stack import MrcImageStack
from careamics.dataset.image_stack_loader import load_mrcs


def test_load_mrcs(tmp_path: Path):
    paths = [tmp_path / "first.mrc", tmp_path / "second.mrc"]
    for path in paths:
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(np.zeros((3, 4, 5), dtype=np.float32))

    image_stacks = load_mrcs(paths, axes="ZYX")

    assert all(isinstance(stack, MrcImageStack) for stack in image_stacks)
    assert [stack.source for stack in image_stacks] == paths
    assert [stack.data_shape for stack in image_stacks] == [
        (1, 1, 3, 4, 5),
        (1, 1, 3, 4, 5),
    ]

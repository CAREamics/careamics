from pathlib import Path

import numpy as np
import tifffile

from careamics.dataset_ng.patch_extractor.image_stack import FileImageStack
from careamics.dataset_ng.patch_extractor.limit_file_extractor import (
    LimitFilesPatchExtractor,
)
from careamics.dataset_ng.patching_strategies import RandomPatchingStrategy


def test_files_limited(tmp_path: Path):
    """Test that `LimitFilesPatchExtractor` never has more than two files loaded."""

    rng = np.random.default_rng(42)

    # set up test data
    data_shapes = [(64, 48), (55, 54), (71, 65), (32, 32)]
    input_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_shapes]
    paths = [tmp_path / f"{i}.tiff" for i in range(len(data_shapes))]
    for path, data in zip(paths, input_data, strict=True):
        tifffile.imwrite(path, data, metadata={"axes": "YX"})

    # set up patch extractor
    image_stacks = [FileImageStack.from_tiff(path, axes="YX") for path in paths]
    patch_extractor = LimitFilesPatchExtractor(image_stacks)

    # using random patching
    patching = RandomPatchingStrategy(
        [image_stack.data_shape for image_stack in image_stacks],
        patch_size=(16, 16),
        seed=42,
    )
    patch_specs = [patching.get_patch_spec(i) for i in range(patching.n_patches)]
    # shuffle the indices because adjacent indices still belong to the same image
    indices = np.arange(len(patch_specs))
    rng.shuffle(indices)

    for i in indices:
        patch_extractor.extract_patch(**patch_specs[i])
        is_loaded = [
            image_stack.is_loaded for image_stack in patch_extractor.image_stacks
        ]
        # 1 or less files should be loaded at any time
        assert np.count_nonzero(is_loaded) <= 1

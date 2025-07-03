from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.dataset_ng.patch_extractor.image_stack import CziImageStack

# skip if fail imports
pylib = pytest.importorskip("pylibCZIrw")

from pylibCZIrw import czi as pyczi  # noqa: E402

T_EXPR = "does not contain a T axis"
Z_EXPR = "does not contain a Z axis"


def create_test_czi(file_path: Path, data: NDArray | list[NDArray]):
    if not isinstance(data, list):
        data = [data]

    with pyczi.create_czi(str(file_path)) as czi:
        xoffs = 0
        for scene_idx, scene_data in enumerate(data):
            for t in range(scene_data.shape[0]):
                for c in range(scene_data.shape[1]):
                    for z in range(scene_data.shape[2]):
                        czi.write(
                            scene_data[t, c, z],
                            plane={"C": c, "T": t, "Z": z},
                            location=(xoffs, 0),
                            scene=scene_idx,
                        )
            xoffs += scene_data.shape[-1] + 20


@pytest.mark.czi
@pytest.mark.parametrize(
    "orig_shape, depth_axis, expected_axes, expected_shape, sample_idx, expect_raise",
    [
        # 2-D Images
        ((1, 1, 1, 32, 48), "none", "SCYX", [1, 1, 32, 48], 0, nullcontext()),
        ((1, 1, 1, 32, 48), "T", "", [], 0, pytest.raises(RuntimeError, match=T_EXPR)),
        ((1, 1, 1, 32, 48), "Z", "", [], 0, pytest.raises(RuntimeError, match=Z_EXPR)),
        # 3-D Volumes
        ((1, 1, 16, 32, 48), "none", "SCYX", [16, 1, 32, 48], 9, nullcontext()),
        ((1, 1, 16, 32, 48), "T", "", [], 9, pytest.raises(RuntimeError, match=T_EXPR)),
        ((1, 1, 16, 32, 48), "Z", "SCZYX", [1, 1, 16, 32, 48], 0, nullcontext()),
        # 2-D Time-Series
        ((8, 1, 1, 32, 48), "none", "SCYX", [8, 1, 32, 48], 3, nullcontext()),
        ((8, 1, 1, 32, 48), "Z", "", [], 3, pytest.raises(RuntimeError, match=Z_EXPR)),
        ((8, 1, 1, 32, 48), "T", "SCTYX", [1, 1, 8, 32, 48], 0, nullcontext()),
        # 3-D Time-Series
        ((8, 1, 16, 32, 48), "none", "SCYX", [8 * 16, 1, 32, 48], 35, nullcontext()),
        ((8, 1, 16, 32, 48), "Z", "SCZYX", [8, 1, 16, 32, 48], 7, nullcontext()),
        ((8, 1, 16, 32, 48), "T", "SCTYX", [16, 1, 8, 32, 48], 12, nullcontext()),
        # Multiple channels
        ((8, 3, 16, 32, 48), "Z", "SCZYX", [8, 3, 16, 32, 48], 2, nullcontext()),
    ],
)
def test_extract_patch(
    tmp_path: Path,
    orig_shape: tuple[int, ...],
    depth_axis: Literal["none", "Z", "T"],
    expected_axes: str,
    expected_shape: list[int],
    sample_idx: int,
    expect_raise,
):
    # reference data to compare against
    data = np.random.randn(*orig_shape).astype(np.float32)

    # save data as a czi file to ininitialise image stack with
    file_path = tmp_path / "test_czi.czi"
    create_test_czi(file_path=file_path, data=data)

    # initialise CziImageStack
    with expect_raise:
        image_stack = CziImageStack(data_path=file_path, depth_axis=depth_axis)

    # stop here if expecting an exception
    if expected_axes == "":
        return

    # check axes and shape
    assert image_stack.axes == expected_axes
    assert image_stack.data_shape == expected_shape

    # test extracted patch matches patch from reference data
    if len(expected_axes) < 5:
        coords = (11, 4)
        patch_size = (16, 9)

        extracted_patch = image_stack.extract_patch(
            sample_idx=sample_idx, coords=coords, patch_size=patch_size
        )

        data_ref = np.moveaxis(data, 2, 1)  # (T, Z, C, Y, X)
        data_ref = data_ref.reshape(-1, *data_ref.shape[-3:])
        patch_ref = data_ref[
            sample_idx,
            :,
            coords[0] : coords[0] + patch_size[0],
            coords[1] : coords[1] + patch_size[1],
        ]
    else:
        coords = (2, 11, 4)
        patch_size = (4, 16, 9)

        extracted_patch = image_stack.extract_patch(
            sample_idx=sample_idx, coords=coords, patch_size=patch_size
        )

        if "T" in expected_axes and expected_axes.index("T") == 2:
            data_ref = data.swapaxes(0, 2)
        else:
            data_ref = data
        patch_ref = data_ref[
            sample_idx,
            ...,
            coords[0] : coords[0] + patch_size[0],
            coords[1] : coords[1] + patch_size[1],
            coords[2] : coords[2] + patch_size[2],
        ]
    np.testing.assert_array_equal(extracted_patch, patch_ref)


@pytest.mark.czi
def test_multiple_scenes(
    tmp_path: Path,
):
    original_shapes = [
        [4, 1, 8, 16, 32],
        [4, 1, 8, 24, 18],
        [4, 1, 8, 45, 32],
    ]

    # reference data to compare against
    data_ref = [np.random.randn(*shape).astype(np.float32) for shape in original_shapes]

    # save data as a czi file to ininitialise image stack with
    file_path = tmp_path / "test_czi.czi"
    create_test_czi(file_path=file_path, data=data_ref)

    # Test reading scene metadata
    scene_rectangles = CziImageStack.get_bounding_rectangles(file_path)
    assert len(scene_rectangles) == len(original_shapes)

    scene_rectangle_sizes = [(rect.h, rect.w) for rect in scene_rectangles.values()]
    expected_rectangle_sizes = [(shape[-2], shape[-1]) for shape in original_shapes]
    assert scene_rectangle_sizes == expected_rectangle_sizes

    # Test reading from individual scenes
    for scene_idx, expected_shape in enumerate(original_shapes):
        image_stack = CziImageStack(
            data_path=file_path, scene=scene_idx, depth_axis="Z"
        )
        assert image_stack.data_shape == expected_shape

        t = expected_shape[0] - scene_idx - 1
        coords = (2, 9, 4)
        patch_size = (4, 7, 13)

        extracted_patch = image_stack.extract_patch(
            sample_idx=t, coords=coords, patch_size=patch_size
        )
        patch_ref = data_ref[scene_idx][
            t,
            :,
            coords[0] : coords[0] + patch_size[0],
            coords[1] : coords[1] + patch_size[1],
            coords[2] : coords[2] + patch_size[2],
        ]
        np.testing.assert_array_equal(extracted_patch, patch_ref)

import numpy as np
import zarr

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    PatchExtractor,
    create_zarr_extractor,
)
from careamics.dataset_ng.patching_strategies import (
    PatchSpecs,
    TileSpecs,
    TilingStrategy,
)
from careamics.dataset_ng.writer.zarr_prediction_callback import (
    ZarrPredictionWriterCallback,
)


def create_image_region(
    axes, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
) -> ImageRegionData:
    data_idx = patch_spec["data_idx"]
    source = extractor.image_stacks[data_idx].source
    chunks = extractor.image_stacks[data_idx].chunks
    return ImageRegionData(
        data=patch,
        source=str(source),
        dtype=str(extractor.image_stacks[data_idx].data_dtype),
        data_shape=extractor.image_stacks[data_idx].data_shape,
        axes=axes,
        region_spec=patch_spec,
        chunks=chunks,
    )


def gen_image_regions(my_patch_extractor: PatchExtractor, my_strategy: TilingStrategy):
    for i in range(my_strategy.n_patches):
        patch_spec: TileSpecs = my_strategy.get_patch_spec(i)
        patch = my_patch_extractor.extract_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )
        region = create_image_region("YX", patch, patch_spec, my_patch_extractor)

        yield region


def test_zarr_prediction_callback_identity(tmp_path):
    # create data
    arrays = np.arange(6 * 32 * 32).reshape((6, 32, 32))

    # write zarr sources to two different zarrs, at different levels
    path = tmp_path / "source.zarr"
    g = zarr.create_group(path)

    image1_group = g.create_group("images1")
    single_array = image1_group.create_array(
        name="single_image",
        data=arrays[0],
        chunks=(32, 32),
    )
    array_uris = [single_array.store_path]  # uris to the arrays

    image2_group = g.create_group("images2")
    for i in range(1, 5):
        array = image2_group.create_array(
            name=f"image_stack_{i}",
            data=arrays[i],
            chunks=(32, 32),
        )
        array_uris.append(array.store_path)

    path2 = tmp_path / "source2.zarr"
    g2 = zarr.create_group(path2)
    array_root = g2.create_array(
        name="root_array",
        data=arrays[5],
        chunks=(32, 32),
    )
    array_uris.append(array_root.store_path)

    # create extractor and tiling strategy
    patch_extractor = create_zarr_extractor(
        source=array_uris,
        axes="YX",
    )

    strategy = TilingStrategy(
        data_shapes=[(1, 1, 32, 32) for _ in range(len(array_uris))],
        tile_size=(8, 8),
        overlaps=(4, 4),
    )
    assert strategy.n_patches == 6 * ((32 - 4) / (8 - 4)) ** 2

    # use callback to write predictions
    callback = ZarrPredictionWriterCallback()
    for img in gen_image_regions(patch_extractor, strategy):
        # assert isinstance(img, ImageRegionData)
        callback.write_on_batch_end(None, None, [img], None, None, None, None)

    # check that the arrays have been writtent correctly to the first zarr
    assert (tmp_path / "source_output.zarr").exists()

    g_output = zarr.open_group(tmp_path / "source_output.zarr", mode="r")
    assert np.array_equal(g_output["images1/single_image"][:], arrays[0])
    for i in range(1, 5):
        assert np.array_equal(g_output[f"images2/image_stack_{i}"][:], arrays[i])

    # check that the array has been written correctly to the second zarr
    assert (tmp_path / "source2_output.zarr").exists()
    g_output2 = zarr.open_group(tmp_path / "source2_output.zarr", mode="r")
    assert np.array_equal(g_output2["root_array"][:], arrays[5])

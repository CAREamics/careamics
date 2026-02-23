import random

import numpy as np
import pytest
from torch.utils.data.dataloader import default_collate

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.lightning.dataset_ng.prediction import (
    combine_samples,
    convert_prediction,
    decollate_image_region_data,
    restore_original_shape,
)


@pytest.fixture
def batches(source_name: str) -> list[ImageRegionData]:
    """
    Fixture providing batches of `ImageRegionData` to test decollate.

    Returns
    -------
    list of ImageRegionData
        List of batches of `ImageRegionData`.
    """
    batches = []
    for b in range(5):  # data idx spread over multiple batches
        batch = []
        for i in range(4):
            data_idx = (b * 4 + i) // 10

            batch.append(
                ImageRegionData(
                    source=f"{data_idx}.tiff" if source_name == "file" else "array",
                    data=(
                        data_idx * np.ones((32, 32)).astype(np.float32)
                    ),  # individual sample, B dim added by collate
                    data_shape=(10, 32, 32),
                    dtype="float32",
                    # axes describes the full dataset shape
                    axes="SYX",
                    # non-sense to check that they are properly decollated
                    region_spec={
                        "data_idx": data_idx,
                        "sample_idx": (b * 4 + i) % 10,
                        "coords": (0, i * 4),
                        "patch_size": (4, i),
                    },
                    additional_metadata={
                        "chunks": (1, 1, 16, 16),
                        "shards": (1, 1, 32, 32),
                    },
                    original_data_shape=(10, 32, 32),
                )
            )

        batch_collated = default_collate(batch)
        batches.append(batch_collated)
    return batches


class TestCombineSamples:
    @pytest.mark.parametrize("source_name", ["file"])  # injected in fixture
    def test_combine_prediction_by_data_idx(
        self, batches: list[ImageRegionData]
    ) -> None:
        """Test `combine_prediction_by_data_idx` function."""
        all_decollated: list[ImageRegionData] = []
        for batch in batches:
            decollated = decollate_image_region_data(batch)
            all_decollated.extend(decollated)

        combined_predictions, _ = combine_samples(all_decollated)
        assert len(combined_predictions) == 2
        assert combined_predictions[0].shape == (10, 32, 32)
        assert np.all(combined_predictions[0] == 0)
        assert combined_predictions[1].shape == (10, 32, 32)
        assert np.all(combined_predictions[1] == 1)

    @pytest.mark.parametrize("source_name", ["file"])  # injected in fixture
    def test_data_idx_order(self, batches: list[ImageRegionData]) -> None:
        """Test that `combine_prediction_by_data_idx` returns data in the correct
        order."""
        all_decollated: list[ImageRegionData] = []
        for batch in batches:
            decollated = decollate_image_region_data(batch)
            all_decollated.extend(decollated)

        data_indices = np.unique([b.region_spec["data_idx"] for b in all_decollated])
        assert len(data_indices) == 2
        assert data_indices[0] < data_indices[1]

        # shuffle decollated to test ordering
        random.shuffle(all_decollated)

        # predict and check ordering
        combined_predictions, sources = combine_samples(all_decollated)
        assert len(combined_predictions) == len(sources) == 2
        assert sources[0] == "0.tiff"
        assert sources[1] == "1.tiff"

        assert np.all(combined_predictions[0] == 0)  # data_idx = 0
        assert np.all(combined_predictions[1] == 1)  # data_idx = 1


@pytest.mark.parametrize("n_batch", [1, 2, 4])
def test_decollate_image_region_data(n_batch) -> None:
    """
    Test `decollate_image_region_data` function.

    Parameters
    ----------
    ordered_array : NDArray
        Ordered array fixture.
    """
    batch: list[ImageRegionData] = []
    for i in range(n_batch):
        batch.append(
            ImageRegionData(
                source="array.tiff",
                # individual sample, B dim is added by collate
                data=np.ones((4, 4)).astype(np.float32),
                data_shape=(32, 32),
                dtype=str(np.float32),
                axes="YX",
                region_spec={
                    "data_idx": i,
                    "sample_idx": 0,
                    "coords": (0, i * 4),
                    "patch_size": (4, 4),
                },
                additional_metadata={
                    "chunks": (1, 1, 16, i),
                    "shards": (1, 1, 32, i * 2),
                },
                original_data_shape=(32, 32),
            )
        )

    batch_collated = default_collate(batch)
    decollated = decollate_image_region_data(batch_collated)

    assert len(decollated) == len(batch)
    for i in range(len(batch)):
        np.testing.assert_array_equal(decollated[i].data, batch[i].data, verbose=True)
        assert decollated[i].source == batch[i].source
        assert decollated[i].data_shape == batch[i].data_shape
        assert decollated[i].dtype == batch[i].dtype
        assert decollated[i].axes == batch[i].axes
        assert decollated[i].region_spec == batch[i].region_spec
        assert (
            decollated[i].additional_metadata["chunks"]
            == batch[i].additional_metadata["chunks"]
        )
        assert (
            decollated[i].additional_metadata["shards"]
            == batch[i].additional_metadata["shards"]
        )


class TestRestoreShape:
    def test_identity(self) -> None:
        """Test that `restore_original_shape` returns the same array if already in
        original shape."""
        original_shape = (5, 3, 32, 32)
        prediction = np.arange(np.prod(original_shape)).reshape(original_shape)
        restored = restore_original_shape(prediction, "SCYX", original_shape)
        assert np.array_equal(restored, prediction)

    def test_singleton_s(self) -> None:
        """Test that `restore_original_shape` removes singleton S axis."""
        prediction = np.arange(3 * 32 * 32).reshape((1, 3, 32, 32))
        restored = restore_original_shape(prediction, "CYX", (3, 32, 32))
        assert np.array_equal(restored, prediction[0])

    def test_singleton_c(self) -> None:
        """Test that `restore_original_shape` removes singleton C axis."""
        prediction = np.arange(5 * 32 * 32).reshape((5, 1, 32, 32))
        restored = restore_original_shape(prediction, "SYX", (5, 32, 32))
        assert np.array_equal(restored, prediction[:, 0])

    def test_unflatten_s_and_t(self) -> None:
        """Test that `restore_original_shape` unflattens S axis to S and T."""
        prediction = np.arange(5 * 3 * 16 * 32 * 32).reshape((15, 1, 16, 32, 32))
        restored = restore_original_shape(prediction, "STZYX", (3, 5, 16, 32, 32))

        shape = restored.shape
        for s in range(shape[0]):
            for t in range(shape[1]):
                np.testing.assert_array_equal(
                    restored[s, t], prediction[s * shape[1] + t, 0]
                )

    def test_s_to_t(self) -> None:
        """Test that `restore_original_shape` converts S axis to T."""
        prediction = np.arange(5 * 32 * 32).reshape((5, 1, 32, 32))
        restored = restore_original_shape(prediction, "TYX", (5, 32, 32))

        shape = restored.shape
        for t in range(shape[0]):
            np.testing.assert_array_equal(restored[t], prediction[t, 0])

    def test_unflatten_s_to_z_and_t(self) -> None:
        """Test that `restore_original_shape` unflattens S axis to Z and T.

        Case encountered with CZI when no depth axis is selected.
        """
        prediction = np.arange(5 * 3 * 32 * 32).reshape((15, 1, 32, 32))
        restored = restore_original_shape(prediction, "TZYX", (3, 5, 32, 32))

        shape = restored.shape
        for z in range(shape[0]):
            for t in range(shape[1]):
                np.testing.assert_array_equal(
                    restored[z, t], prediction[z * shape[1] + t, 0]
                )

    def test_z_to_t(self) -> None:
        """Test that `restore_original_shape` converts Z axis to T.

        Case encountered with CZI when T is selected as depth axis.
        """
        prediction = np.arange(15 * 32 * 32).reshape((1, 1, 15, 32, 32))
        restored = restore_original_shape(prediction, "TYX", (15, 32, 32))

        shape = restored.shape
        for t in range(shape[0]):
            np.testing.assert_array_equal(restored[t], prediction[0, 0, t])

    def test_reorder_axes(self) -> None:
        """Test that `restore_original_shape` reorders axes to match original."""
        prediction = np.arange(5 * 3 * 16 * 32 * 32).reshape((5, 3, 16, 32, 32))
        restored = restore_original_shape(prediction, "CYXZS", (3, 32, 32, 16, 5))

        shape = restored.shape
        for c in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    for z in range(shape[3]):
                        for s in range(shape[4]):
                            np.testing.assert_array_equal(
                                restored[c, y, x, z, s], prediction[s, c, z, y, x]
                            )


class TestConvertPrediction:
    @pytest.mark.parametrize("source_name", ["file"])  # injected in fixture
    def test_convert_arrays(self, batches: list[ImageRegionData]) -> None:
        """Test `convert_arrays` function."""
        predictions, sources = convert_prediction(batches, tiled=False)
        assert len(predictions) == 2  # 2 data idx in fixture
        assert predictions[0].shape == (10, 32, 32)
        assert predictions[1].shape == (10, 32, 32)
        assert sources == ["0.tiff", "1.tiff"]

    @pytest.mark.parametrize("source_name", ["array"])  # injected in fixture
    def test_convert_arrays_empty_list(self, batches: list[ImageRegionData]) -> None:
        """Test `convert_arrays` with "array" sources returns an empty list."""
        _, sources = convert_prediction(batches, tiled=False)
        assert sources == []

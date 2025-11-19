import numpy as np
import pytest
from torch.utils.data.dataloader import default_collate

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.lightning.dataset_ng.prediction import (
    combine_samples,
    decollate_image_region_data,
)


@pytest.fixture
def batches() -> list[ImageRegionData]:
    """
    Fixture providing batches of `ImageRegionData`.

    Returns
    -------
    list of ImageRegionData
        List of batches of `ImageRegionData`.
    """
    batches = []
    for b in range(5):  # data idx spread over multiple batches
        batch = []
        for i in range(4):
            batch.append(
                ImageRegionData(
                    source="array.tiff",
                    data=np.ones((1, 32, 32)).astype(
                        np.float32
                    ),  # B dim added by collate
                    data_shape=(10, 32, 32),
                    dtype="float32",
                    axes="SYX",
                    # non-sense to check that they are properly decollated
                    region_spec={
                        "data_idx": (b * 4 + i) // 10,
                        "sample_idx": (b * 4 + i) % 10,
                        "coords": (0, i * 4),
                        "patch_size": (4, i),
                    },
                    chunks=(1, 1, 8, i * 8),
                )
            )

        batch_collated = default_collate(batch)
        batches.append(batch_collated)
    return batches


def test_combine_prediction_by_data_idx(batches: list[ImageRegionData]) -> None:
    """Test `combine_prediction_by_data_idx` function."""
    all_decollated: list[ImageRegionData] = []
    for batch in batches:
        decollated = decollate_image_region_data(batch)
        all_decollated.extend(decollated)

    combined_predictions, _ = combine_samples(all_decollated)
    assert len(combined_predictions) == 2
    assert combined_predictions[0].shape == (10, 32, 32)
    assert combined_predictions[1].shape == (10, 32, 32)


@pytest.mark.parametrize("n_batch", [1, 2, 4])
def test_decollate_image_region_data(n_batch) -> None:
    """
    Test `decollate_image_region_data` function.

    Parameters
    ----------
    ordered_array : NDArray
        Ordered array fixture.
    """
    batch = []
    for i in range(n_batch):
        batch.append(
            ImageRegionData(
                source="array.tiff",
                data=np.ones((1, 4, 4)).astype(np.float32),  # B dim is added by collate
                data_shape=(32, 32),
                dtype=str(np.float32),
                axes="YX",
                region_spec={
                    "data_idx": i,
                    "sample_idx": 0,
                    "coords": (0, i * 4),
                    "patch_size": (4, 4),
                },
                chunks=(1, 1, 8, 8),
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

import numpy as np
import pytest
import tifffile

from careamics.dataset.extraction_strategy import ExtractionStrategy
from careamics.dataset.tiff_dataset import TiffDataset


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((16, 16), "YX"),
        ((8, 16, 16), "ZYX"),
        ((2, 16, 16), "SYX"),
    ],
)
def test_tiff_dataset(tmp_path, ordered_array, shape, axes):
    """Test loading tiffs."""
    array = ordered_array(shape)
    array2 = array * 2

    # save arrays
    tifffile.imwrite(tmp_path / "test1.tif", array)
    tifffile.imwrite(tmp_path / "test2.tif", array2)

    # create dataset
    dataset = TiffDataset(
        data_path=tmp_path,
        data_format="tif",
        axes=axes,
        patch_extraction_method=ExtractionStrategy.SEQUENTIAL,
    )

    # check mean and std
    all_arrays = np.concatenate([array, array2], axis=0)
    mean = np.mean(all_arrays)
    std = np.mean([np.std(array), np.std(array2)])

    assert dataset.mean == pytest.approx(mean)
    assert dataset.std == pytest.approx(std)


def test_tiff_dataset_not_dir(tmp_path, ordered_array):
    """Test loading tiffs."""
    array = ordered_array((16, 16))

    # save array
    tifffile.imwrite(tmp_path / "test1.tif", array)

    # create dataset
    with pytest.raises(ValueError):
        TiffDataset(
            data_path=tmp_path / "test1.tif",
            data_format="tif",
            axes="YX",
            patch_extraction_method=ExtractionStrategy.SEQUENTIAL,
        )

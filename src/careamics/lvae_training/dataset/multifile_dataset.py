from collections.abc import Sequence
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from .config import DatasetConfig
from .lc_dataset import LCMultiChDloader
from .multich_dataset import MultiChDloader
from .types import DataSplitType


class TwoChannelData(Sequence):
    """
    each element in data_arr should be a N*H*W array
    """

    def __init__(self, data_arr1, data_arr2, paths_data1=None, paths_data2=None):
        assert len(data_arr1) == len(data_arr2)
        self.paths1 = paths_data1
        self.paths2 = paths_data2

        self._data = []
        for i in range(len(data_arr1)):
            assert data_arr1[i].shape == data_arr2[i].shape
            assert (
                len(data_arr1[i].shape) == 3
            ), f"Each element in data arrays should be a N*H*W, but {data_arr1[i].shape}"
            self._data.append(
                np.concatenate(
                    [data_arr1[i][..., None], data_arr2[i][..., None]], axis=-1
                )
            )

    def __len__(self):
        n = 0
        for x in self._data:
            n += x.shape[0]
        return n

    def __getitem__(self, idx):
        n = 0
        for dataidx, x in enumerate(self._data):
            if idx < n + x.shape[0]:
                if self.paths1 is None:
                    return x[idx - n], None
                else:
                    return x[idx - n], (self.paths1[dataidx], self.paths2[dataidx])
            n += x.shape[0]
        raise IndexError("Index out of range")


class MultiChannelData(Sequence):
    """
    each element in data_arr should be a N*H*W array
    """

    def __init__(self, data_arr, paths=None):
        self.paths = paths

        self._data = data_arr

    def __len__(self):
        n = 0
        for x in self._data:
            n += x.shape[0]
        return n

    def __getitem__(self, idx):
        n = 0
        for dataidx, x in enumerate(self._data):
            if idx < n + x.shape[0]:
                if self.paths is None:
                    return x[idx - n], None
                else:
                    return x[idx - n], (self.paths[dataidx])
            n += x.shape[0]
        raise IndexError("Index out of range")


class SingleFileLCDset(LCMultiChDloader):
    def __init__(
        self,
        preloaded_data: NDArray,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
    ):
        self._preloaded_data = preloaded_data
        super().__init__(
            data_config,
            fpath,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    @property
    def data_path(self):
        return self._fpath

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        pass

    def load_data(
        self,
        data_config: DatasetConfig,
        datasplit_type: DataSplitType,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
        allow_generation=None,
    ):
        self._data = self._preloaded_data
        assert "channel_1" not in data_config or isinstance(data_config.channel_1, str)
        assert "channel_2" not in data_config or isinstance(data_config.channel_2, str)
        assert "channel_3" not in data_config or isinstance(data_config.channel_3, str)
        self._loaded_data_preprocessing(data_config)


class SingleFileDset(MultiChDloader):
    def __init__(
        self,
        preloaded_data: NDArray,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
    ):
        self._preloaded_data = preloaded_data
        super().__init__(
            data_config,
            fpath,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        pass

    @property
    def data_path(self):
        return self._fpath

    def load_data(
        self,
        data_config: DatasetConfig,
        datasplit_type: DataSplitType,
        load_data_fn: Callable[..., NDArray],
        val_fraction=None,
        test_fraction=None,
        allow_generation=None,
    ):
        self._data = self._preloaded_data
        assert (
            "channel_1" not in data_config
        ), "Outdated config file. Please remove channel_1, channel_2, channel_3 from the config file."
        assert (
            "channel_2" not in data_config
        ), "Outdated config file. Please remove channel_1, channel_2, channel_3 from the config file."
        assert (
            "channel_3" not in data_config
        ), "Outdated config file. Please remove channel_1, channel_2, channel_3 from the config file."
        self._loaded_data_preprocessing(data_config)


class MultiFileDset:
    """
    Here, we handle dataset having multiple files. Each file can have a different spatial dimension and number of frames (Z stack).
    """

    def __init__(
        self,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable[..., Union[TwoChannelData, MultiChannelData]],
        val_fraction=None,
        test_fraction=None,
    ):
        self._fpath = fpath
        data: Union[TwoChannelData, MultiChannelData] = load_data_fn(
            data_config,
            self._fpath,
            data_config.datasplit_type,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        self.dsets = []

        for i in range(len(data)):
            prefetched_data, fpath_tuple = data[i]
            if (
                data_config.multiscale_lowres_count is not None
                and data_config.multiscale_lowres_count > 1
            ):

                self.dsets.append(
                    SingleFileLCDset(
                        prefetched_data[None],
                        data_config,
                        fpath_tuple,
                        load_data_fn,
                        val_fraction=val_fraction,
                        test_fraction=test_fraction,
                    )
                )

            else:
                self.dsets.append(
                    SingleFileDset(
                        prefetched_data[None],
                        data_config,
                        fpath_tuple,
                        load_data_fn,
                        val_fraction=val_fraction,
                        test_fraction=test_fraction,
                    )
                )

        self.rm_bkground_set_max_val_and_upperclip_data(
            data_config.max_val, data_config.datasplit_type
        )
        count = 0
        avg_height = 0
        avg_width = 0
        for dset in self.dsets:
            shape = dset.get_data_shape()
            avg_height += shape[1]
            avg_width += shape[2]
            count += shape[0]

        avg_height = int(avg_height / len(self.dsets))
        avg_width = int(avg_width / len(self.dsets))
        print(
            f"{self.__class__.__name__} avg height: {avg_height}, avg width: {avg_width}, count: {count}"
        )

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        self.set_max_val(max_val, datasplit_type)
        self.upperclip_data()

    def set_mean_std(self, mean_val, std_val):
        for dset in self.dsets:
            dset.set_mean_std(mean_val, std_val)

    def get_mean_std(self):
        return self.dsets[0].get_mean_std()

    def compute_max_val(self):
        max_val_arr = []
        for dset in self.dsets:
            max_val_arr.append(dset.compute_max_val())
        return np.max(max_val_arr)

    def set_max_val(self, max_val, datasplit_type):
        if datasplit_type == DataSplitType.Train:
            assert max_val is None
            max_val = self.compute_max_val()
        for dset in self.dsets:
            dset.set_max_val(max_val, datasplit_type)

    def upperclip_data(self):
        for dset in self.dsets:
            dset.upperclip_data()

    def get_max_val(self):
        return self.dsets[0].get_max_val()

    def get_img_sz(self):
        return self.dsets[0].get_img_sz()

    def set_img_sz(self, image_size, grid_size):
        for dset in self.dsets:
            dset.set_img_sz(image_size, grid_size)

    def compute_mean_std(self):
        cur_mean = {"target": 0, "input": 0}
        cur_std = {"target": 0, "input": 0}
        for dset in self.dsets:
            mean, std = dset.compute_mean_std()
            cur_mean["target"] += mean["target"]
            cur_mean["input"] += mean["input"]

            cur_std["target"] += std["target"]
            cur_std["input"] += std["input"]

        cur_mean["target"] /= len(self.dsets)
        cur_mean["input"] /= len(self.dsets)
        cur_std["target"] /= len(self.dsets)
        cur_std["input"] /= len(self.dsets)
        return cur_mean, cur_std

    def compute_individual_mean_std(self):
        cum_mean = 0
        cum_std = 0
        for dset in self.dsets:
            mean, std = dset.compute_individual_mean_std()
            cum_mean += mean
            cum_std += std
        return cum_mean / len(self.dsets), cum_std / len(self.dsets)

    def get_num_frames(self):
        return len(self.dsets)

    def reduce_data(
        self, t_list=None, h_start=None, h_end=None, w_start=None, w_end=None
    ):
        assert h_start is None
        assert h_end is None
        assert w_start is None
        assert w_end is None
        self.dsets = [self.dsets[t] for t in t_list]
        print(
            f"[{self.__class__.__name__}] Data reduced. New data count: {len(self.dsets)}"
        )

    def __len__(self):
        out = 0
        for dset in self.dsets:
            out += len(dset)
        return out

    def __getitem__(self, idx):
        cum_len = 0
        for dset in self.dsets:
            cum_len += len(dset)
            if idx < cum_len:
                rel_idx = idx - (cum_len - len(dset))
                return dset[rel_idx]

        raise IndexError("Index out of range")

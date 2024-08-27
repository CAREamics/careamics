"""
A place for Datasets and Dataloaders.
"""

from typing import Tuple, Union
import os

import numpy as np

# import albumentations as A
import ml_collections
from skimage.transform import resize

from .data_utils import (
    DataSplitType,
    DataType,
    GridAlignement,
    GridIndexManager,
    IndexSwitcher,
    load_tiff,
    get_datasplit_tuples,
    get_mrc_data,
)


def get_train_val_data(
    data_config,
    fpath,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    allow_generation=None,
    ignore_specific_datapoints=None,
):
    """
    Load the data from the given path and split them in training, validation and test sets.

    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    if data_config.data_type == DataType.SeparateTiffData:
        fpath1 = os.path.join(fpath, data_config.ch1_fname)
        fpath2 = os.path.join(fpath, data_config.ch2_fname)
        fpaths = [fpath1, fpath2]
        fpath0 = ""
        if "ch_input_fname" in data_config:
            fpath0 = os.path.join(fpath, data_config.ch_input_fname)
            fpaths = [fpath0] + fpaths

        print(
            f"Loading from {fpath} Channels: "
            f"{fpath1},{fpath2}, inp:{fpath0} Mode:{DataSplitType.name(datasplit_type)}"
        )

        data = np.concatenate([load_tiff(fpath)[..., None] for fpath in fpaths], axis=3)
        if data_config.data_type == DataType.PredictedTiffData:
            assert len(data.shape) == 5 and data.shape[-1] == 1
            data = data[..., 0].copy()
        # data = data[::3].copy()
        # NOTE: This was not the correct way to do it. It is so because the noise present in the input was directly related
        # to the noise present in the channels and so this is not the way we would get the data.
        # We need to add the noise independently to the input and the target.

        # if data_config.get('poisson_noise_factor', False):
        #     data = np.random.poisson(data)
        # if data_config.get('enable_gaussian_noise', False):
        #     synthetic_scale = data_config.get('synthetic_gaussian_scale', 0.1)
        #     print('Adding Gaussian noise with scale', synthetic_scale)
        #     noise = np.random.normal(0, synthetic_scale, data.shape)
        #     data = data + noise

        if datasplit_type == DataSplitType.All:
            return data.astype(np.float32)

        train_idx, val_idx, test_idx = get_datasplit_tuples(
            val_fraction, test_fraction, len(data), starting_test=True
        )
        if datasplit_type == DataSplitType.Train:
            return data[train_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Val:
            return data[val_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Test:
            return data[test_idx].astype(np.float32)

    elif data_config.data_type == DataType.BioSR_MRC:
        num_channels = data_config.get("num_channels", 2)
        fpaths = []
        data_list = []
        for i in range(num_channels):
            fpath1 = os.path.join(fpath, data_config.get(f"ch{i + 1}_fname"))
            fpaths.append(fpath1)
            data = get_mrc_data(fpath1)[..., None]
            data_list.append(data)

        dirname = os.path.dirname(os.path.dirname(fpaths[0])) + "/"

        msg = ",".join([x[len(dirname) :] for x in fpaths])
        print(
            f"Loaded from {dirname} Channels:{len(fpaths)} {msg} Mode:{DataSplitType.name(datasplit_type)}"
        )
        N = data_list[0].shape[0]
        for data in data_list:
            N = min(N, data.shape[0])

        cropped_data = []
        for data in data_list:
            cropped_data.append(data[:N])

        data = np.concatenate(cropped_data, axis=3)

        if datasplit_type == DataSplitType.All:
            return data.astype(np.float32)

        train_idx, val_idx, test_idx = get_datasplit_tuples(
            val_fraction, test_fraction, len(data), starting_test=True
        )
        if datasplit_type == DataSplitType.Train:
            return data[train_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Val:
            return data[val_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Test:
            return data[test_idx].astype(np.float32)


class MultiChDloader:

    def __init__(
        self,
        data_config: ml_collections.ConfigDict,
        fpath: str,
        datasplit_type: DataSplitType = None,
        val_fraction: float = None,
        test_fraction: float = None,
        normalized_input=None,
        enable_rotation_aug: bool = False,
        enable_random_cropping: bool = False,
        use_one_mu_std=None,
        allow_generation: bool = False,
        max_val: float = None,
        grid_alignment=GridAlignement.LeftTop,
        overlapping_padding_kwargs=None,
        print_vars: bool = True,
    ):
        """
        Here, an image is split into grids of size img_sz.
        Args:
            repeat_factor: Since we are doing a random crop, repeat_factor is
                given which can repeatedly sample from the same image. If self.N=12
                and repeat_factor is 5, then index upto 12*5 = 60 is allowed.
            use_one_mu_std: If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different meean and stdev are used.

        """
        self._data_type = data_config.data_type
        self._fpath = fpath
        self._data = self.N = self._noise_data = None

        # Hardcoded params, not included in the config file.

        # by default, if the noise is present, add it to the input and target.
        self._disable_noise = False  # to add synthetic noise
        self._train_index_switcher = None
        # NOTE: Input is the sum of the different channels. It is not the average of the different channels.
        self._input_is_sum = data_config.get("input_is_sum", False)
        self._num_channels = data_config.get("num_channels", 2)
        self._input_idx = data_config.get("input_idx", None)
        self._tar_idx_list = data_config.get("target_idx_list", None)

        if datasplit_type == DataSplitType.Train:
            self._datausage_fraction = 1.0
            # assert self._datausage_fraction == 1.0, 'Not supported. Use validtarget_random_fraction and training_validtarget_fraction to get the same effect'
            self._validtarget_rand_fract = None
            # self._validtarget_random_fraction_final = data_config.get('validtarget_random_fraction_final', None)
            # self._validtarget_random_fraction_stepepoch = data_config.get('validtarget_random_fraction_stepepoch', None)
            # self._idx_count = 0
        elif datasplit_type == DataSplitType.Val:
            self._datausage_fraction = 1.0
        else:
            self._datausage_fraction = 1.0

        self.load_data(
            data_config,
            datasplit_type,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            allow_generation=allow_generation,
        )
        self._normalized_input = normalized_input
        self._quantile = 1.0
        self._channelwise_quantile = False
        self._background_quantile = 0.0
        self._clip_background_noise_to_zero = False
        self._skip_normalization_using_mean = False
        self._empty_patch_replacement_enabled = False

        self._background_values = None

        self._grid_alignment = grid_alignment
        self._overlapping_padding_kwargs = overlapping_padding_kwargs
        if self._grid_alignment == GridAlignement.LeftTop:
            assert (
                self._overlapping_padding_kwargs is None
                or data_config.multiscale_lowres_count is not None
            ), "Padding is not used with this alignement style"
        elif self._grid_alignment == GridAlignement.Center:
            assert (
                self._overlapping_padding_kwargs is not None
            ), "With Center grid alignment, padding is needed."

        self._is_train = datasplit_type == DataSplitType.Train

        # input = alpha * ch1 + (1-alpha)*ch2.
        # alpha is sampled randomly between these two extremes
        self._start_alpha_arr = self._end_alpha_arr = self._return_alpha = (
            self._alpha_weighted_target
        ) = None

        self._img_sz = self._grid_sz = self._repeat_factor = self.idx_manager = None
        if self._is_train:
            self._start_alpha_arr = None
            self._end_alpha_arr = None
            self._alpha_weighted_target = False

            self.set_img_sz(
                data_config.image_size,
                (
                    data_config.grid_size
                    if "grid_size" in data_config
                    else data_config.image_size
                ),
            )

            # if self._validtarget_rand_fract is not None:
            #     self._train_index_switcher = IndexSwitcher(self.idx_manager, data_config, self._img_sz)
            #     self._std_background_arr = None

        else:
            self.set_img_sz(
                data_config.image_size,
                (
                    data_config.grid_size
                    if "grid_size" in data_config
                    else data_config.image_size
                ),
            )

        self._return_alpha = False
        self._return_index = False

        # self._empty_patch_replacement_enabled = data_config.get("empty_patch_replacement_enabled",
        #                                                         False) and self._is_train
        # if self._empty_patch_replacement_enabled:
        #     self._empty_patch_replacement_channel_idx = data_config.empty_patch_replacement_channel_idx
        #     self._empty_patch_replacement_probab = data_config.empty_patch_replacement_probab
        #     data_frames = self._data[..., self._empty_patch_replacement_channel_idx]
        #     # NOTE: This is on the raw data. So, it must be called before removing the background.
        #     self._empty_patch_fetcher = EmptyPatchFetcher(self.idx_manager,
        #                                                   self._img_sz,
        #                                                   data_frames,
        #                                                   max_val_threshold=data_config.empty_patch_max_val_threshold)

        self.rm_bkground_set_max_val_and_upperclip_data(max_val, datasplit_type)

        # For overlapping dloader, image_size and repeat_factors are not related. hence a different function.

        self._mean = None
        self._std = None
        self._use_one_mu_std = use_one_mu_std
        # Hardcoded
        self._target_separate_normalization = True

        self._enable_rotation = enable_rotation_aug
        self._enable_random_cropping = enable_random_cropping
        self._uncorrelated_channels = (
            data_config.get("uncorrelated_channels", False) and self._is_train
        )
        assert self._is_train or self._uncorrelated_channels is False
        assert (
            self._enable_random_cropping is True or self._uncorrelated_channels is False
        )
        # Randomly rotate [-90,90]

        self._rotation_transform = None
        if self._enable_rotation:
            raise NotImplementedError(
                "Augmentation by means of rotation is not supported yet."
            )
            self._rotation_transform = A.Compose([A.Flip(), A.RandomRotate90()])

        if print_vars:
            msg = self._init_msg()
            print(msg)

    def disable_noise(self):
        assert (
            self._poisson_noise_factor is None
        ), "This is not supported. Poisson noise is added to the data itself and so the noise cannot be disabled."
        self._disable_noise = True

    def enable_noise(self):
        self._disable_noise = False

    def get_data_shape(self):
        return self._data.shape

    def load_data(
        self,
        data_config,
        datasplit_type,
        val_fraction=None,
        test_fraction=None,
        allow_generation=None,
    ):
        self._data = get_train_val_data(
            data_config,
            self._fpath,
            datasplit_type,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            allow_generation=allow_generation,
        )

        old_shape = self._data.shape
        if self._datausage_fraction < 1.0:
            framepixelcount = np.prod(self._data.shape[1:3])
            pixelcount = int(
                len(self._data) * framepixelcount * self._datausage_fraction
            )
            frame_count = int(np.ceil(pixelcount / framepixelcount))
            last_frame_reduced_size, _ = IndexSwitcher.get_reduced_frame_size(
                self._data.shape[:3], self._datausage_fraction
            )
            self._data = self._data[:frame_count].copy()
            if frame_count == 1:
                self._data = self._data[
                    :, :last_frame_reduced_size, :last_frame_reduced_size
                ].copy()
            print(
                f"[{self.__class__.__name__}] New data shape: {self._data.shape} Old: {old_shape}"
            )

        msg = ""
        if data_config.get("poisson_noise_factor", -1) > 0:
            self._poisson_noise_factor = data_config.poisson_noise_factor
            msg += f"Adding Poisson noise with factor {self._poisson_noise_factor}.\t"
            self._data = (
                np.random.poisson(self._data / self._poisson_noise_factor)
                * self._poisson_noise_factor
            )

        if data_config.get("enable_gaussian_noise", False):
            synthetic_scale = data_config.get("synthetic_gaussian_scale", 0.1)
            msg += f"Adding Gaussian noise with scale {synthetic_scale}"
            # 0 => noise for input. 1: => noise for all targets.
            shape = self._data.shape
            self._noise_data = np.random.normal(
                0, synthetic_scale, (*shape[:-1], shape[-1] + 1)
            )
            if data_config.get("input_has_dependant_noise", False):
                msg += ". Moreover, input has dependent noise"
                self._noise_data[..., 0] = np.mean(self._noise_data[..., 1:], axis=-1)
        print(msg)

        self.N = len(self._data)
        assert (
            self._data.shape[-1] == self._num_channels
        ), "Number of channels in data and config do not match."

    def save_background(self, channel_idx, frame_idx, background_value):
        self._background_values[frame_idx, channel_idx] = background_value

    def get_background(self, channel_idx, frame_idx):
        return self._background_values[frame_idx, channel_idx]

    def remove_background(self):

        self._background_values = np.zeros((self._data.shape[0], self._data.shape[-1]))

        if self._background_quantile == 0.0:
            assert (
                self._clip_background_noise_to_zero is False
            ), "This operation currently happens later in this function."
            return

        if self._data.dtype in [np.uint16]:
            # unsigned integer creates havoc
            self._data = self._data.astype(np.int32)

        for ch in range(self._data.shape[-1]):
            for idx in range(self._data.shape[0]):
                qval = np.quantile(self._data[idx, ..., ch], self._background_quantile)
                assert (
                    np.abs(qval) > 20
                ), "We are truncating the qval to an integer which will only make sense if it is large enough"
                # NOTE: Here, there can be an issue if you work with normalized data
                qval = int(qval)
                self.save_background(ch, idx, qval)
                self._data[idx, ..., ch] -= qval

        if self._clip_background_noise_to_zero:
            self._data[self._data < 0] = 0

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        self.remove_background()
        self.set_max_val(max_val, datasplit_type)
        self.upperclip_data()

    def upperclip_data(self):
        if isinstance(self.max_val, list):
            chN = self._data.shape[-1]
            assert chN == len(self.max_val)
            for ch in range(chN):
                ch_data = self._data[..., ch]
                ch_q = self.max_val[ch]
                ch_data[ch_data > ch_q] = ch_q
                self._data[..., ch] = ch_data
        else:
            self._data[self._data > self.max_val] = self.max_val

    def compute_max_val(self):
        if self._channelwise_quantile:
            max_val_arr = [
                np.quantile(self._data[..., i], self._quantile)
                for i in range(self._data.shape[-1])
            ]
            return max_val_arr
        else:
            return np.quantile(self._data, self._quantile)

    def set_max_val(self, max_val, datasplit_type):

        if max_val is None:
            assert datasplit_type == DataSplitType.Train
            self.max_val = self.compute_max_val()
        else:
            assert max_val is not None
            self.max_val = max_val

    def get_max_val(self):
        return self.max_val

    def get_img_sz(self):
        return self._img_sz

    def reduce_data(
        self, t_list=None, h_start=None, h_end=None, w_start=None, w_end=None
    ):
        if t_list is None:
            t_list = list(range(self._data.shape[0]))
        if h_start is None:
            h_start = 0
        if h_end is None:
            h_end = self._data.shape[1]
        if w_start is None:
            w_start = 0
        if w_end is None:
            w_end = self._data.shape[2]

        self._data = self._data[t_list, h_start:h_end, w_start:w_end, :].copy()
        if self._noise_data is not None:
            self._noise_data = self._noise_data[
                t_list, h_start:h_end, w_start:w_end, :
            ].copy()

        self.N = len(t_list)
        self.set_img_sz(self._img_sz, self._grid_sz)
        print(
            f"[{self.__class__.__name__}] Data reduced. New data shape: {self._data.shape}"
        )

    def set_img_sz(self, image_size, grid_size):
        """
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """

        self._img_sz = image_size
        self._grid_sz = grid_size
        self.idx_manager = GridIndexManager(
            self._data.shape, self._grid_sz, self._img_sz, self._grid_alignment
        )
        self.set_repeat_factor()

    def set_repeat_factor(self):
        if self._grid_sz > 1:
            self._repeat_factor = self.idx_manager.grid_rows(
                self._grid_sz
            ) * self.idx_manager.grid_cols(self._grid_sz)
        else:
            self._repeat_factor = self.idx_manager.grid_rows(
                self._img_sz
            ) * self.idx_manager.grid_cols(self._img_sz)

    def _init_msg(
        self,
    ):
        msg = (
            f"[{self.__class__.__name__}] Train:{int(self._is_train)} Sz:{self._img_sz}"
        )
        msg += f" N:{self.N} NumPatchPerN:{self._repeat_factor}"
        # msg += f' NormInp:{self._normalized_input}'
        # msg += f' SingleNorm:{self._use_one_mu_std}'
        msg += f" Rot:{self._enable_rotation}"
        msg += f" RandCrop:{self._enable_random_cropping}"
        msg += f" Channel:{self._num_channels}"
        # msg += f' Q:{self._quantile}'
        if self._input_is_sum:
            msg += f" SummedInput:{self._input_is_sum}"

        if self._empty_patch_replacement_enabled:
            msg += f" ReplaceWithRandSample:{self._empty_patch_replacement_enabled}"
        if self._uncorrelated_channels:
            msg += f" Uncorr:{self._uncorrelated_channels}"
        if self._empty_patch_replacement_enabled:
            msg += f"-{self._empty_patch_replacement_channel_idx}-{self._empty_patch_replacement_probab}"
        if self._background_quantile > 0.0:
            msg += f" BckQ:{self._background_quantile}"

        if self._start_alpha_arr is not None:
            msg += f" Alpha:[{self._start_alpha_arr},{self._end_alpha_arr}]"
        return msg

    def _crop_imgs(self, index, *img_tuples: np.ndarray):
        h, w = img_tuples[0].shape[-2:]
        if self._img_sz is None:
            return (
                *img_tuples,
                {"h": [0, h], "w": [0, w], "hflip": False, "wflip": False},
            )

        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

        cropped_imgs = []
        for img in img_tuples:
            img = self._crop_flip_img(img, h_start, w_start, False, False)
            cropped_imgs.append(img)

        return (
            *tuple(cropped_imgs),
            {
                "h": [h_start, h_start + self._img_sz],
                "w": [w_start, w_start + self._img_sz],
                "hflip": False,
                "wflip": False,
            },
        )

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        if self._grid_alignment == GridAlignement.LeftTop:
            # In training, this is used.
            # NOTE: It is my opinion that if I just use self._crop_img_with_padding, it will work perfectly fine.
            # The only benefit this if else loop provides is that it makes it easier to see what happens during training.
            new_img = img[
                ..., h_start : h_start + self._img_sz, w_start : w_start + self._img_sz
            ]
            return new_img
        elif self._grid_alignment == GridAlignement.Center:
            # During evaluation, this is used. In this situation, we can have negative h_start, w_start. Or h_start +self._img_sz can be larger than frame
            # In these situations, we need some sort of padding. This is not needed  in the LeftTop alignement.
            return self._crop_img_with_padding(img, h_start, w_start)

    def get_begin_end_padding(self, start_pos, max_len):
        """
        The effect is that the image with size self._grid_sz is in the center of the patch with sufficient
        padding on all four sides so that the final patch size is self._img_sz.
        """
        pad_start = 0
        pad_end = 0
        if start_pos < 0:
            pad_start = -1 * start_pos

        pad_end = max(0, start_pos + self._img_sz - max_len)

        return pad_start, pad_end

    def _crop_img_with_padding(self, img: np.ndarray, h_start: int, w_start: int):
        _, H, W = img.shape
        h_on_boundary = self.on_boundary(h_start, H)
        w_on_boundary = self.on_boundary(w_start, W)

        assert h_start < H
        assert w_start < W

        assert h_start + self._img_sz <= H or h_on_boundary
        assert w_start + self._img_sz <= W or w_on_boundary
        # max() is needed since h_start could be negative.
        new_img = img[
            ...,
            max(0, h_start) : h_start + self._img_sz,
            max(0, w_start) : w_start + self._img_sz,
        ]
        padding = np.array([[0, 0], [0, 0], [0, 0]])

        if h_on_boundary:
            pad = self.get_begin_end_padding(h_start, H)
            padding[1] = pad
        if w_on_boundary:
            pad = self.get_begin_end_padding(w_start, W)
            padding[2] = pad

        if not np.all(padding == 0):
            new_img = np.pad(new_img, padding, **self._overlapping_padding_kwargs)

        return new_img

    def _crop_flip_img(
        self, img: np.ndarray, h_start: int, w_start: int, h_flip: bool, w_flip: bool
    ):
        new_img = self._crop_img(img, h_start, w_start)
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def __len__(self):
        return self.N * self._repeat_factor

    def _load_img(
        self, index: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the channels and also the respective noise channels.
        """
        if isinstance(index, int) or isinstance(index, np.int64):
            idx = index
        else:
            idx = index[0]

        imgs = self._data[self.idx_manager.get_t(idx)]
        loaded_imgs = [imgs[None, ..., i] for i in range(imgs.shape[-1])]
        noise = []
        if self._noise_data is not None and not self._disable_noise:
            noise = [
                self._noise_data[self.idx_manager.get_t(idx)][None, ..., i]
                for i in range(self._noise_data.shape[-1])
            ]
        return tuple(loaded_imgs), tuple(noise)

    def get_mean_std(self):
        return self._mean, self._std

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

    def normalize_img(self, *img_tuples):
        mean, std = self.get_mean_std()
        mean = mean["target"]
        std = std["target"]
        mean = mean.squeeze()
        std = std.squeeze()
        normalized_imgs = []
        for i, img in enumerate(img_tuples):
            img = (img - mean[i]) / std[i]
            normalized_imgs.append(img)
        return tuple(normalized_imgs)
    
    def normalize_input(self, x):
        mean_dict, std_dict = self.get_mean_std()
        mean_ = mean_dict["input"].mean()
        std_ = std_dict["input"].mean()
        return (x - mean_) / std_

    def normalize_target(self, target):
        mean_dict, std_dict = self.get_mean_std()
        mean_ = mean_dict["target"].squeeze(0)
        std_ = std_dict["target"].squeeze(0)
        return (target - mean_) / std_

    def get_grid_size(self):
        return self._grid_sz

    def get_idx_manager(self):
        return self.idx_manager

    def per_side_overlap_pixelcount(self):
        return (self._img_sz - self._grid_sz) // 2

    def on_boundary(self, cur_loc, frame_size):
        return cur_loc + self._img_sz > frame_size or cur_loc < 0

    def _get_deterministic_hw(self, index: Union[int, Tuple[int, int]]):
        """
        It returns the top-left corner of the patch corresponding to index.
        """
        if isinstance(index, int) or isinstance(index, np.int64):
            idx = index
            grid_size = self._grid_sz
        else:
            idx, grid_size = index

        h_start, w_start = self.idx_manager.get_deterministic_hw(
            idx, grid_size=grid_size
        )
        if self._grid_alignment == GridAlignement.LeftTop:
            return h_start, w_start
        elif self._grid_alignment == GridAlignement.Center:
            pad = self.per_side_overlap_pixelcount()
            return h_start - pad, w_start - pad

    def compute_individual_mean_std(self):
        # numpy 1.19.2 has issues in computing for large arrays. https://github.com/numpy/numpy/issues/8869
        # mean = np.mean(self._data, axis=(0, 1, 2))
        # std = np.std(self._data, axis=(0, 1, 2))
        mean_arr = []
        std_arr = []
        for ch_idx in range(self._data.shape[-1]):
            mean_ = (
                0.0
                if self._skip_normalization_using_mean
                else self._data[..., ch_idx].mean()
            )
            if self._noise_data is not None:
                std_ = (
                    self._data[..., ch_idx] + self._noise_data[..., ch_idx + 1]
                ).std()
            else:
                std_ = self._data[..., ch_idx].std()

            mean_arr.append(mean_)
            std_arr.append(std_)

        mean = np.array(mean_arr)
        std = np.array(std_arr)

        return mean[None, :, None, None], std[None, :, None, None]

    def compute_mean_std(self, allow_for_validation_data=False):
        """
        Note that we must compute this only for training data.
        """
        assert (
            self._is_train is True or allow_for_validation_data
        ), "This is just allowed for training data"
        assert self._use_one_mu_std is True, "This is the only supported case"

        if self._input_idx is not None:
            assert (
                self._tar_idx_list is not None
            ), "tar_idx_list must be set if input_idx is set."
            assert self._noise_data is None, "This is not supported with noise"
            assert (
                self._target_separate_normalization is True
            ), "This is not supported with target_separate_normalization=False"

            mean, std = self.compute_individual_mean_std()
            mean_dict = {
                "input": mean[:, self._input_idx : self._input_idx + 1],
                "target": mean[:, self._tar_idx_list],
            }
            std_dict = {
                "input": std[:, self._input_idx : self._input_idx + 1],
                "target": std[:, self._tar_idx_list],
            }
            return mean_dict, std_dict

        if self._input_is_sum:
            assert self._noise_data is None, "This is not supported with noise"
            mean = [
                np.mean(self._data[..., k : k + 1], keepdims=True)
                for k in range(self._num_channels)
            ]
            mean = np.sum(mean, keepdims=True)[0]
            std = np.linalg.norm(
                [
                    np.std(self._data[..., k : k + 1], keepdims=True)
                    for k in range(self._num_channels)
                ],
                keepdims=True,
            )[0]
        else:
            # mean&std are computed for all the channels at the same time
            mean = np.mean(self._data, keepdims=True).reshape(1, 1, 1, 1)
            if self._noise_data is not None:
                std = np.std(
                    self._data + self._noise_data[..., 1:], keepdims=True
                ).reshape(1, 1, 1, 1)
            else:
                std = np.std(self._data, keepdims=True).reshape(1, 1, 1, 1)

        mean = np.repeat(mean, self._num_channels, axis=1)
        std = np.repeat(std, self._num_channels, axis=1)

        if self._skip_normalization_using_mean:
            mean = np.zeros_like(mean)

        mean_dict = {"input": mean}  # , 'target':mean}
        std_dict = {"input": std}  # , 'target':std}

        if self._target_separate_normalization: # True
            # this is computed for each channel separately
            mean, std = self.compute_individual_mean_std()

        mean_dict["target"] = mean
        std_dict["target"] = std
        return mean_dict, std_dict

    def _get_random_hw(self, h: int, w: int):
        """
        Random starting position for the crop for the img with index `index`.
        """
        if h != self._img_sz:
            h_start = np.random.choice(h - self._img_sz)
            w_start = np.random.choice(w - self._img_sz)
        else:
            h_start = 0
            w_start = 0
        return h_start, w_start

    def _get_img(self, index: Union[int, Tuple[int, int]]):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img_tuples, noise_tuples = self._load_img(index)
        cropped_img_tuples = self._crop_imgs(index, *img_tuples, *noise_tuples)[:-1]
        cropped_noise_tuples = cropped_img_tuples[len(img_tuples) :]
        cropped_img_tuples = cropped_img_tuples[: len(img_tuples)]
        return cropped_img_tuples, cropped_noise_tuples

    def replace_with_empty_patch(self, img_tuples):
        empty_index = self._empty_patch_fetcher.sample()
        empty_img_tuples = self._get_img(empty_index)
        final_img_tuples = []
        for tuple_idx in range(len(img_tuples)):
            if tuple_idx == self._empty_patch_replacement_channel_idx:
                final_img_tuples.append(empty_img_tuples[tuple_idx])
            else:
                final_img_tuples.append(img_tuples[tuple_idx])
        return tuple(final_img_tuples)

    def get_mean_std_for_input(self):
        mean, std = self.get_mean_std()
        return mean["input"], std["input"]

    def _compute_target(self, img_tuples, alpha):
        if self._tar_idx_list is not None and isinstance(self._tar_idx_list, int):
            target = img_tuples[self._tar_idx_list]
        else:
            if self._tar_idx_list is not None:
                assert isinstance(self._tar_idx_list, list) or isinstance(
                    self._tar_idx_list, tuple
                )
                img_tuples = [img_tuples[i] for i in self._tar_idx_list]

            if self._alpha_weighted_target:
                assert self._input_is_sum is False
                target = []
                for i in range(len(img_tuples)):
                    target.append(img_tuples[i] * alpha[i])
                target = np.concatenate(target, axis=0)
            else:
                target = np.concatenate(img_tuples, axis=0)
        return target

    def _compute_input_with_alpha(self, img_tuples, alpha_list):
        # assert self._normalized_input is True, "normalization should happen here"
        if self._input_idx is not None:
            inp = img_tuples[self._input_idx]
        else:
            inp = 0
            for alpha, img in zip(alpha_list, img_tuples):
                inp += img * alpha

            if self._normalized_input is False:
                return inp.astype(np.float32)

        mean, std = self.get_mean_std_for_input()
        mean = mean.squeeze()
        std = std.squeeze()
        if mean.size == 1:
            mean = mean.reshape(
                1,
            )
            std = std.reshape(
                1,
            )

        for i in range(len(mean)):
            assert mean[0] == mean[i]
            assert std[0] == std[i]

        inp = (inp - mean[0]) / std[0]
        return inp.astype(np.float32)

    def _sample_alpha(self):
        alpha_arr = []
        for i in range(self._num_channels):
            alpha_pos = np.random.rand()
            alpha = self._start_alpha_arr[i] + alpha_pos * (
                self._end_alpha_arr[i] - self._start_alpha_arr[i]
            )
            alpha_arr.append(alpha)
        return alpha_arr

    def _compute_input(self, img_tuples):
        alpha = [1 / len(img_tuples) for _ in range(len(img_tuples))]
        if self._start_alpha_arr is not None:
            alpha = self._sample_alpha()

        inp = self._compute_input_with_alpha(img_tuples, alpha)
        if self._input_is_sum:
            inp = len(img_tuples) * inp
        return inp, alpha

    def _get_index_from_valid_target_logic(self, index):
        if self._validtarget_rand_fract is not None:
            if np.random.rand() < self._validtarget_rand_fract:
                index = self._train_index_switcher.get_valid_target_index()
            else:
                index = self._train_index_switcher.get_invalid_target_index()
        return index

    def _rotate(self, img_tuples, noise_tuples):
        return self._rotate2D(img_tuples, noise_tuples)

    def _rotate2D(self, img_tuples, noise_tuples):
        img_kwargs = {}
        for i, img in enumerate(img_tuples):
            for k in range(len(img)):
                img_kwargs[f"img{i}_{k}"] = img[k]

        noise_kwargs = {}
        for i, nimg in enumerate(noise_tuples):
            for k in range(len(nimg)):
                noise_kwargs[f"noise{i}_{k}"] = nimg[k]

        keys = list(img_kwargs.keys()) + list(noise_kwargs.keys())
        self._rotation_transform.add_targets({k: "image" for k in keys})
        rot_dic = self._rotation_transform(
            image=img_tuples[0][0], **img_kwargs, **noise_kwargs
        )
        rotated_img_tuples = []
        for i, img in enumerate(img_tuples):
            if len(img) == 1:
                rotated_img_tuples.append(rot_dic[f"img{i}_0"][None])
            else:
                rotated_img_tuples.append(
                    np.concatenate(
                        [rot_dic[f"img{i}_{k}"][None] for k in range(len(img))], axis=0
                    )
                )

        rotated_noise_tuples = []
        for i, nimg in enumerate(noise_tuples):
            if len(nimg) == 1:
                rotated_noise_tuples.append(rot_dic[f"noise{i}_0"][None])
            else:
                rotated_noise_tuples.append(
                    np.concatenate(
                        [rot_dic[f"noise{i}_{k}"][None] for k in range(len(nimg))],
                        axis=0,
                    )
                )

        return rotated_img_tuples, rotated_noise_tuples

    def get_uncorrelated_img_tuples(self, index):
        img_tuples, noise_tuples = self._get_img(index)
        assert len(noise_tuples) == 0
        img_tuples = [img_tuples[0]]
        for ch_idx in range(1, len(img_tuples)):
            new_index = np.random.randint(len(self))
            other_img_tuples, _ = self._get_img(new_index)
            img_tuples.append(other_img_tuples[ch_idx])
        return img_tuples, noise_tuples

    def __getitem__(
        self, index: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._train_index_switcher is not None:
            index = self._get_index_from_valid_target_logic(index)

        if self._uncorrelated_channels:
            img_tuples, noise_tuples = self.get_uncorrelated_img_tuples(index)
        else:
            img_tuples, noise_tuples = self._get_img(index)

        assert (
            self._empty_patch_replacement_enabled != True
        ), "This is not supported with noise"

        if self._empty_patch_replacement_enabled:
            if np.random.rand() < self._empty_patch_replacement_probab:
                img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # add noise to input
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = [x + noise_tuples[0] * factor for x in img_tuples]
        else:
            input_tuples = img_tuples
        inp, alpha = self._compute_input(input_tuples)

        # add noise to target.
        if len(noise_tuples) >= 1:
            img_tuples = [x + noise for x, noise in zip(img_tuples, noise_tuples[1:])]

        target = self._compute_target(img_tuples, alpha)

        # normalize
        norm_target = self.normalize_target(target)

        return inp, norm_target

        # if self._return_alpha:
        #     output.append(alpha)

        # if self._return_index:
        #     output.append(index)

        # if isinstance(index, int) or isinstance(index, np.int64):
        #     return tuple(output)

        # _, grid_size = index
        # output.append(grid_size)
        # return tuple(output)


class LCMultiChDloader(MultiChDloader):

    def __init__(
        self,
        data_config,
        fpath: str,
        datasplit_type: DataSplitType = None,
        val_fraction=None,
        test_fraction=None,
        normalized_input=None,
        enable_rotation_aug: bool = False,
        use_one_mu_std=None,
        num_scales: int = None,
        enable_random_cropping=False,
        padding_kwargs: dict = None,
        allow_generation: bool = False,
        lowres_supervision=None,
        max_val=None,
        grid_alignment=GridAlignement.LeftTop,
        overlapping_padding_kwargs=None,
        print_vars=True,
    ):
        """
        Args:
            num_scales: The number of resolutions at which we want the input. Note that the target is formed at the
                        highest resolution.
        """
        self._padding_kwargs = (
            padding_kwargs  # mode=padding_mode, constant_values=constant_value
        )
        if overlapping_padding_kwargs is not None:
            assert (
                self._padding_kwargs == overlapping_padding_kwargs
            ), "During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None"

        else:
            overlapping_padding_kwargs = padding_kwargs

        super().__init__(
            data_config,
            fpath,
            datasplit_type=datasplit_type,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            normalized_input=normalized_input,
            enable_rotation_aug=enable_rotation_aug,
            enable_random_cropping=enable_random_cropping,
            use_one_mu_std=use_one_mu_std,
            allow_generation=allow_generation,
            max_val=max_val,
            grid_alignment=grid_alignment,
            overlapping_padding_kwargs=overlapping_padding_kwargs,
            print_vars=print_vars,
        )
        self.num_scales = num_scales
        assert self.num_scales is not None
        self._scaled_data = [self._data]
        self._scaled_noise_data = [self._noise_data]

        assert isinstance(self.num_scales, int) and self.num_scales >= 1
        self._lowres_supervision = lowres_supervision
        assert isinstance(self._padding_kwargs, dict)
        assert "mode" in self._padding_kwargs

        for _ in range(1, self.num_scales):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            ds_data = resize(
                self._scaled_data[-1].astype(np.float32), new_shape
            ).astype(self._scaled_data[-1].dtype)
            # NOTE: These asserts are important. the resize method expects np.float32. otherwise, one gets weird results.
            assert (
                ds_data.max() / self._scaled_data[-1].max() < 5
            ), "Downsampled image should not have very different values"
            assert (
                ds_data.max() / self._scaled_data[-1].max() > 0.2
            ), "Downsampled image should not have very different values"

            self._scaled_data.append(ds_data)
            # do the same for noise
            if self._noise_data is not None:
                noise_data = resize(self._scaled_noise_data[-1], new_shape)
                self._scaled_noise_data.append(noise_data)

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f" Pad:{self._padding_kwargs}"
        return msg

    def _load_scaled_img(
        self, scaled_index, index: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index
        imgs = self._scaled_data[scaled_index][idx % self.N]
        imgs = tuple([imgs[None, :, :, i] for i in range(imgs.shape[-1])])
        if self._noise_data is not None:
            noisedata = self._scaled_noise_data[scaled_index][idx % self.N]
            noise = tuple(
                [noisedata[None, :, :, i] for i in range(noisedata.shape[-1])]
            )
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            # since we are using this lowres images for just the input, we need to add the noise of the input.
            assert self._lowres_supervision is None or self._lowres_supervision is False
            imgs = tuple([img + noise[0] * factor for img in imgs])
        return imgs

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        return self._crop_img_with_padding(img, h_start, w_start)

    def _get_img(self, index: int):
        """
        Returns the primary patch along with low resolution patches centered on the primary patch.
        """
        img_tuples, noise_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

        cropped_img_tuples = [
            self._crop_flip_img(img, h_start, w_start, False, False)
            for img in img_tuples
        ]
        cropped_noise_tuples = [
            self._crop_flip_img(noise, h_start, w_start, False, False)
            for noise in noise_tuples
        ]
        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {
            i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))
        }
        for scale_idx in range(1, self.num_scales):
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2

            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2

            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, h_start, w_start, False, False)
                for img in scaled_img_tuples
            ]
            for ch_idx in range(len(img_tuples)):
                allres_versions[ch_idx].append(scaled_cropped_img_tuples[ch_idx])

        output_img_tuples = tuple(
            [
                np.concatenate(allres_versions[ch_idx])
                for ch_idx in range(len(img_tuples))
            ]
        )
        return output_img_tuples, cropped_noise_tuples

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if self._uncorrelated_channels:
            img_tuples, noise_tuples = self.get_uncorrelated_img_tuples(index)
        else:
            img_tuples, noise_tuples = self._get_img(index)

        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        assert self._lowres_supervision != True
        # add noise to input
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = []
            for x in img_tuples:
                # NOTE: other LC levels already have noise added. So, we just need to add noise to the highest resolution.
                x[0] = x[0] + noise_tuples[0] * factor
                input_tuples.append(x)
        else:
            input_tuples = img_tuples

        inp, alpha = self._compute_input(input_tuples)
        # assert self._alpha_weighted_target in [False, None]
        target_tuples = [img[:1] for img in img_tuples]
        # add noise to target.
        if len(noise_tuples) >= 1:
            target_tuples = [
                x + noise for x, noise in zip(target_tuples, noise_tuples[1:])
            ]

        target = self._compute_target(target_tuples, alpha)

        # normalize
        norm_target = self.normalize_target(target)

        return inp, norm_target

        # if self._return_alpha:
        #     output.append(alpha)

        # if isinstance(index, int):
        #     return tuple(output)

        # _, grid_size = index
        # output.append(grid_size)
        # return tuple(output)

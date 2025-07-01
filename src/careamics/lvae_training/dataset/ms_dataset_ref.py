"""
A place for Datasets and Dataloaders.
"""

from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Callable, Union

import numpy as np
from skimage.transform import resize

from .config import DatasetConfig
from .types import DataSplitType, TilingMode
from .utils.empty_patch_fetcher import EmptyPatchFetcher
from .utils.index_manager import GridIndexManagerRef


class MultiChDloaderRef:
    def __init__(
        self,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction: float = None,
        test_fraction: float = None,
    ):
        """ """
        self._data_type = data_config.data_type
        self._fpath = Path(fpath)
        self._data = None
        self._3Ddata = False  # TODO wtf it was 5D
        self._tiling_mode = data_config.tiling_mode
        # by default, if the noise is present, add it to the input and target.
        self._depth3D = data_config.depth3D
        self._mode_3D = data_config.mode_3D
        # NOTE: Input is the sum of the different channels. It is not the average of the different channels.
        self._input_is_sum = data_config.input_is_sum
        self._num_channels = data_config.num_channels
        self._input_idx = data_config.input_idx
        self._tar_idx_list = data_config.target_idx_list

        self.load_data(
            data_config,
            data_config.datasplit_type,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            allow_generation=data_config.allow_generation,
        )

        self._data_shapes = self.get_data_shapes()
        self._normalized_input = data_config.normalized_input
        self._quantile = 1.0
        self._channelwise_quantile = False
        self._background_quantile = 0.0
        self._clip_background_noise_to_zero = False
        self._skip_normalization_using_mean = False
        self._empty_patch_replacement_enabled = False

        self._background_values = None

        self._overlapping_padding_kwargs = data_config.overlapping_padding_kwargs
        if self._tiling_mode in [TilingMode.TrimBoundary, TilingMode.ShiftBoundary]:
            if (
                self._overlapping_padding_kwargs is None
                or data_config.multiscale_lowres_count is not None
            ):
                # raise warning
                print("Padding is not used with this alignement style")
        else:
            assert (
                self._overlapping_padding_kwargs is not None
            ), "When not trimming boudnary, padding is needed."

        self._is_train = data_config.datasplit_type == DataSplitType.Train

        # input = alpha * ch1 + (1-alpha)*ch2.
        # alpha is sampled randomly between these two extremes
        self._start_alpha_arr = self._end_alpha_arr = self._return_alpha = None

        self._img_sz = self._grid_sz = self._repeat_factor = self.idx_manager = None

        # changed set_img_sz because "grid_size" in data_config returns false
        try:
            grid_size = data_config.grid_size
        except AttributeError:
            grid_size = data_config.image_size

        if self._is_train:
            self._start_alpha_arr = data_config.start_alpha  # TODO why only for train?
            self._end_alpha_arr = data_config.end_alpha

        self.set_img_sz(data_config.image_size, grid_size)

        self._empty_patch_replacement_enabled = (
            data_config.empty_patch_replacement_enabled and self._is_train
        )
        if self._empty_patch_replacement_enabled:
            self._empty_patch_replacement_channel_idx = (
                data_config.empty_patch_replacement_channel_idx
            )
            self._empty_patch_replacement_probab = (
                data_config.empty_patch_replacement_probab
            )
            data_frames = self._data[..., self._empty_patch_replacement_channel_idx]
            # NOTE: This is on the raw data. So, it must be called before removing the background.
            self._empty_patch_fetcher = EmptyPatchFetcher(
                self.idx_manager,
                self._img_sz,
                data_frames,
                max_val_threshold=data_config.empty_patch_max_val_threshold,
            )

        self.rm_bkground_set_max_val_and_upperclip_data(
            data_config.max_val, data_config.datasplit_type
        )

        # For overlapping dloader, image_size and repeat_factors are not related. hence a different function.

        self._mean = None
        self._std = None
        self._use_one_mu_std = data_config.use_one_mu_std

        self._target_separate_normalization = data_config.target_separate_normalization

        self._enable_rotation = data_config.enable_rotation_aug
        flipz_3D = data_config.random_flip_z_3D
        self._flipz_3D = flipz_3D and self._enable_rotation

        self._enable_random_cropping = data_config.enable_random_cropping
        self._uncorrelated_channels = (
            data_config.uncorrelated_channels and self._is_train
        )
        self._uncorrelated_channel_probab = data_config.uncorrelated_channel_probab
        assert self._is_train or self._uncorrelated_channels is False
        assert (
            self._enable_random_cropping is True or self._uncorrelated_channels is False
        )
        # Randomly rotate [-90,90]

        self._rotation_transform = None
        if self._enable_rotation:
            # TODO: fix this import
            import albumentations as A

            self._rotation_transform = A.Compose([A.Flip(), A.RandomRotate90()])

        # TODO: remove print log messages
        # if print_vars:
        #     msg = self._init_msg()
        #     print(msg)

    def get_data_shapes(self):
        if self._3Ddata:  # TODO we assume images don't have a channel dimension
            [
                [
                    im.shape if len(im.shape) == 4 else (1, *im.shape)
                    for im in self._data[ch]
                ]
                for ch in range(len(self._data))
            ]
        else:
            return [
                [
                    im.shape if len(im.shape) == 3 else (1, *im.shape)
                    for im in self._data[ch]
                ]
                for ch in range(len(self._data))
            ]

    def load_data(
        self,
        data_config,
        datasplit_type,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
        allow_generation=None,
    ):
        self._data = load_data_fn(
            data_config,
            self._fpath,
            datasplit_type,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            allow_generation=allow_generation,
        )

    # TODO check for 2D/3D data consistency with config
    # TODO check number of channels consistency with config

    def save_background(self, channel_idx, frame_idx, background_value):
        self._background_values[frame_idx, channel_idx] = background_value

    def get_background(self, channel_idx, frame_idx):
        return self._background_values[frame_idx, channel_idx]

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        # self.remove_background() # TODO revisit
        self.set_max_val(max_val, datasplit_type)
        self.upperclip_data()

    def upperclip_data(self):
        for ch_idx, data in enumerate(self._data):
            if self.max_val[ch_idx] is not None:
                for idx in range(len(data)):
                    data[idx][data[idx] > self.max_val[ch_idx]] = self.max_val[ch_idx]

    def compute_max_val(self):
        # TODO add channelwise quantile ?
        return [
            max([np.quantile(im, self._quantile) for im in ch]) for ch in self._data
        ]

    def set_max_val(self, max_val, datasplit_type):
        if max_val is None:
            assert datasplit_type in [DataSplitType.Train, DataSplitType.All]
            self.max_val = self.compute_max_val()
        else:
            assert max_val is not None
            self.max_val = max_val

    def get_max_val(self):
        return self.max_val

    def get_img_sz(self):
        return self._img_sz

    def get_num_frames(self):
        """Returns the number of the longest channel."""
        return max(self.idx_manager.total_grid_count()[0])

    def reduce_data(
        self,
        t_list=None,
        z_start=None,
        z_end=None,
        h_start=None,
        h_end=None,
        w_start=None,
        w_end=None,
    ):
        raise NotImplementedError("Not implemented")

    def get_idx_manager_shapes(
        self, patch_size: int, grid_size: Union[int, tuple[int, int, int]]
    ):
        numC = len(self._data_shapes)
        if self._3Ddata:
            patch_shape = (1, self._depth3D, patch_size, patch_size)
            if isinstance(grid_size, int):
                grid_shape = (1, 1, grid_size, grid_size)
            else:
                assert len(grid_size) == 3
                assert all(
                    [g <= p for g, p in zip(grid_size, patch_shape[1:-1])]
                ), f"Grid size {grid_size} must be less than patch size {patch_shape[1:-1]}"
                grid_shape = (1, grid_size[0], grid_size[1], grid_size[2])
        else:
            assert isinstance(grid_size, int)
            grid_shape = (1, grid_size, grid_size)
            patch_shape = (1, patch_size, patch_size)

        return patch_shape, grid_shape

    def set_img_sz(self, image_size, grid_size: Union[int, tuple[int, int, int]]):
        """
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """
        # hacky way to deal with image shape from new conf
        self._img_sz = image_size[-1]  # TODO revisit!
        self._grid_sz = grid_size
        shapes = self._data_shapes

        patch_shape, grid_shape = self.get_idx_manager_shapes(
            self._img_sz, self._grid_sz
        )
        self.idx_manager = GridIndexManagerRef(
            shapes, grid_shape, patch_shape, self._tiling_mode
        )

    def __len__(self):
        # If channel length is not equal, return the longest
        return max(self.idx_manager.total_grid_count()[0])

    def _init_msg(
        self,
    ):
        msg = (
            f"[{self.__class__.__name__}] Train:{int(self._is_train)} Sz:{self._img_sz}"
        )
        dim_sizes = [
            self.idx_manager.get_individual_dim_grid_count(dim)
            for dim in range(len(self._data.shape))
        ]
        dim_sizes = ",".join([str(x) for x in dim_sizes])
        msg += f" N:{self.N} NumPatchPerN:{self._repeat_factor}"
        msg += f"{self.idx_manager.total_grid_count()} DimSz:({dim_sizes})"
        msg += f" TrimB:{self._tiling_mode}"
        # msg += f' NormInp:{self._normalized_input}'
        # msg += f' SingleNorm:{self._use_one_mu_std}'
        msg += f" Rot:{self._enable_rotation}"
        if self._flipz_3D:
            msg += f" FlipZ:{self._flipz_3D}"

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

    def _crop_imgs(self, ch_idx: int, patch_idx: int, img: np.ndarray):
        h, w = img.shape[-2:]
        if self._img_sz is None:
            return (
                img,
                {"h": [0, h], "w": [0, w], "hflip": False, "wflip": False},
            )

        if self._enable_random_cropping:
            # this parameter is ambiguous. It toggles between random/deterministic patching
            patch_start_loc = self._get_random_hw(h, w)
            if self._3Ddata:
                patch_start_loc = (
                    np.random.choice(1 + img.shape[-3] - self._depth3D),
                ) + patch_start_loc
        else:
            # Patch coordinates are calculated by the index manager.
            patch_start_loc = self._get_deterministic_loc(ch_idx, patch_idx)
        cropped_img = self._crop_flip_img(img, patch_start_loc, False, False)

        return cropped_img

    def _crop_img(self, img: np.ndarray, patch_start_loc: tuple):
        if self._tiling_mode in [TilingMode.TrimBoundary, TilingMode.ShiftBoundary]:
            # In training, this is used.
            # NOTE: It is my opinion that if I just use self._crop_img_with_padding, it will work perfectly fine.
            # The only benefit this if else loop provides is that it makes it easier to see what happens during training.
            patch_end_loc = (
                np.array(patch_start_loc, dtype=np.int32)
                + self.idx_manager.patch_shape[1:-1]
            )
            if self._3Ddata:
                z_start, h_start, w_start = patch_start_loc
                z_end, h_end, w_end = patch_end_loc
                new_img = img[..., z_start:z_end, h_start:h_end, w_start:w_end]
            else:
                h_start, w_start = patch_start_loc
                h_end, w_end = patch_end_loc
                new_img = img[..., h_start:h_end, w_start:w_end]

            return new_img
        else:
            # During evaluation, this is used. In this situation, we can have negative h_start, w_start. Or h_start +self._img_sz can be larger than frame
            # In these situations, we need some sort of padding. This is not needed  in the LeftTop alignement.
            return self._crop_img_with_padding(img, patch_start_loc)

    def get_begin_end_padding(self, start_pos, end_pos, max_len):
        """
        The effect is that the image with size self._grid_sz is in the center of the patch with sufficient
        padding on all four sides so that the final patch size is self._img_sz.
        """
        pad_start = 0
        pad_end = 0
        if start_pos < 0:
            pad_start = -1 * start_pos

        pad_end = max(0, end_pos - max_len)

        return pad_start, pad_end

    def _crop_img_with_padding(
        self, img: np.ndarray, patch_start_loc, max_len_vals=None
    ):
        if max_len_vals is None:
            max_len_vals = self.idx_manager.data_shape[1:-1]
        patch_end_loc = np.array(patch_start_loc, dtype=int) + np.array(
            self.idx_manager.patch_shape[1:-1], dtype=int
        )
        boundary_crossed = []
        valid_slice = []
        padding = [[0, 0]]
        for start_idx, end_idx, max_len in zip(
            patch_start_loc, patch_end_loc, max_len_vals
        ):
            boundary_crossed.append(end_idx > max_len or start_idx < 0)
            valid_slice.append((max(0, start_idx), min(max_len, end_idx)))
            pad = [0, 0]
            if boundary_crossed[-1]:
                pad = self.get_begin_end_padding(start_idx, end_idx, max_len)
            padding.append(pad)
        # max() is needed since h_start could be negative.
        if self._3Ddata:
            new_img = img[
                ...,
                valid_slice[0][0] : valid_slice[0][1],
                valid_slice[1][0] : valid_slice[1][1],
                valid_slice[2][0] : valid_slice[2][1],
            ]
        else:
            new_img = img[
                ...,
                valid_slice[0][0] : valid_slice[0][1],
                valid_slice[1][0] : valid_slice[1][1],
            ]

        # print(np.array(padding).shape, img.shape, new_img.shape)
        # print(padding)
        if not np.all(padding == 0):
            new_img = np.pad(new_img, padding, **self._overlapping_padding_kwargs)

        return new_img

    def _crop_flip_img(
        self, img: np.ndarray, patch_start_loc: tuple, h_flip: bool, w_flip: bool
    ):
        new_img = self._crop_img(img, patch_start_loc)
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def _load_img(self, ch_idx: int, patch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the channels and also the respective noise channels.
        """
        patch_loc_list = self.idx_manager.get_patch_location_from_patch_idx(
            ch_idx, patch_idx
        )
        # TODO we should be adding channel dim here probably
        img = self._data[ch_idx][patch_loc_list[0]]
        return img

    def get_mean_std(self):
        return self._mean, self._std

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

    def normalize_target(self, target):
        mean_dict, std_dict = self.get_mean_std()
        mean_ = mean_dict["target"]  # .squeeze(0)
        std_ = std_dict["target"]  # .squeeze(0)
        return (target - mean_) / std_

    def get_grid_size(self):
        return self._grid_sz

    def get_idx_manager(self):
        return self.idx_manager

    def per_side_overlap_pixelcount(self):
        return (self._img_sz - self._grid_sz) // 2

    def _get_deterministic_loc(self, ch_idx: int, patch_idx: int):
        """
        It returns the top-left corner of the patch corresponding to index.
        """
        loc_list = self.idx_manager.get_patch_location_from_patch_idx(ch_idx, patch_idx)
        # last dim is channel. we need to take the third and the second last element.
        return loc_list[2:]

    @cache
    def crop_probablities(self, ch_idx):
        sizes = np.array([np.prod(x.shape) for x in self._data[ch_idx]])
        return sizes / sizes.sum()

    def sample_crop(self, ch_idx):
        idx = None
        count = 0
        while idx is None:
            count += 1
            idx = np.random.choice(
                len(self._data[ch_idx]), p=self.crop_probablities(ch_idx)
            )
            data = self._data[ch_idx][idx]  # TODO no channel and S dim ?
            # changed for ndim
            if all(
                d >= self._img_sz for d in data.shape[-2:]
            ):  # TODO dims were hardcoded
                h = np.random.randint(0, data.shape[-2] - self._img_sz)
                w = np.random.randint(0, data.shape[-1] - self._img_sz)

                if len(data.shape) > 2 and not self._3Ddata:
                    s = np.random.randint(0, data.shape[0] - 1)
                    return data[s, h : h + self._img_sz, w : w + self._img_sz]
                else:
                    return data[h : h + self._img_sz, w : w + self._img_sz]

            elif count > 100:
                raise ValueError("Cannot find a valid crop")
            else:
                idx = None

        return None

    def _l2(self, x):
        return np.sqrt(np.mean(np.array(x) ** 2))

    def compute_mean_std(self, allow_for_validation_data=False):
        """
        Note that we must compute this only for training data.
        """
        if self._3Ddata:
            raise NotImplementedError("Not implemented for 3D data")

        if self._input_is_sum:
            mean_tar_dict = defaultdict(list)
            std_tar_dict = defaultdict(list)
            mean_inp = []
            std_inp = []
            for _ in range(30000):
                crops = []
                for ch_idx in range(len(self._data)):
                    crop = self.sample_crop(ch_idx)
                    mean_tar_dict[ch_idx].append(np.mean(crop))
                    std_tar_dict[ch_idx].append(np.std(crop))
                    crops.append(crop)

                inp = 0
                for img in crops:
                    inp += img

                mean_inp.append(np.mean(inp))
                std_inp.append(np.std(inp))

            output_mean = defaultdict(list)
            output_std = defaultdict(list)

            NC = len(self._data)
            for ch_idx in range(NC):
                output_mean["target"].append(np.mean(mean_tar_dict[ch_idx]))
                output_std["target"].append(self._l2(std_tar_dict[ch_idx]))

            output_mean["target"] = np.array(output_mean["target"]).reshape(NC, 1, 1)
            output_std["target"] = np.array(output_std["target"]).reshape(NC, 1, 1)

            output_mean["input"] = np.array([np.mean(mean_inp)]).reshape(1, 1, 1)
            output_std["input"] = np.array([self._l2(std_inp)]).reshape(1, 1, 1)
        else:
            raise NotImplementedError("Not implemented for non-summed input")

        return dict(output_mean), dict(output_std)

    def set_mean_std(self, mean_dict, std_dict):
        self._data_mean = mean_dict
        self._data_std = std_dict

    def get_mean_std(self):
        return self._data_mean, self._data_std

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

    def replace_with_empty_patch(self, img_tuples):
        """
        Replaces the content of one of the channels with background
        """
        empty_index = self._empty_patch_fetcher.sample()
        empty_img_tuples, empty_img_noise_tuples = self._get_img(empty_index)
        assert (
            len(empty_img_noise_tuples) == 0
        ), "Noise is not supported with empty patch replacement"
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

            target = np.stack(img_tuples, axis=0)
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

        # TODO instead we add channel here
        if len(inp.shape) == 2 or (len(inp.shape) == 3 and self._3Ddata):
            inp = inp[None, ...]

        return inp, alpha

    def _get_index_from_valid_target_logic(self, index):
        if self._validtarget_rand_fract is not None:
            if np.random.rand() < self._validtarget_rand_fract:
                index = self._train_index_switcher.get_valid_target_index()
            else:
                index = self._train_index_switcher.get_invalid_target_index()
        return index

    def _rotate2D(self, img_tuples):
        img_kwargs = {}
        for i, img in enumerate(img_tuples):
            for k in range(len(img)):
                img_kwargs[f"img{i}_{k}"] = img[k]

        keys = list(img_kwargs.keys())
        self._rotation_transform.add_targets({k: "image" for k in keys})
        rot_dic = self._rotation_transform(image=img_tuples[0][0], **img_kwargs)

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

        return rotated_img_tuples

    def _rotate3D(self, img_tuples):
        img_kwargs = {}
        # random flip in z direction
        flip_z = self._flipz_3D and np.random.rand() < 0.5
        for i, img in enumerate(img_tuples):
            for j in range(self._depth3D):
                for k in range(len(img)):
                    if flip_z:
                        z_idx = self._depth3D - 1 - j
                    else:
                        z_idx = j
                    img_kwargs[f"img{i}_{z_idx}_{k}"] = img[k, j]

        keys = list(img_kwargs.keys())
        self._rotation_transform.add_targets({k: "image" for k in keys})
        rot_dic = self._rotation_transform(image=img_tuples[0][0][0], **img_kwargs)
        rotated_img_tuples = []
        for i, img in enumerate(img_tuples):
            if len(img) == 1:
                rotated_img_tuples.append(
                    np.concatenate(
                        [
                            rot_dic[f"img{i}_{j}_0"][None, None]
                            for j in range(self._depth3D)
                        ],
                        axis=1,
                    )
                )
            else:
                temp_arr = []
                for k in range(len(img)):
                    temp_arr.append(
                        np.concatenate(
                            [
                                rot_dic[f"img{i}_{j}_{k}"][None, None]
                                for j in range(self._depth3D)
                            ],
                            axis=1,
                        )
                    )
                rotated_img_tuples.append(np.concatenate(temp_arr, axis=0))

        return rotated_img_tuples

    def _rotate(self, img_tuples, noise_tuples):

        if self._3Ddata:
            return self._rotate3D(img_tuples, noise_tuples)
        else:
            return self._rotate2D(img_tuples, noise_tuples)

    def _get_img(self, ch_idx: int, patch_idx: int):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img = self._load_img(ch_idx, patch_idx)
        cropped_img = self._crop_imgs(ch_idx, patch_idx, img)
        return cropped_img

    def get_uncorrelated_img_tuples(self, index):
        """
        Content of channels like actin and nuclei is "correlated" in its
        respective location, this function allows to pick channels' content
        from different patches of the image to make it "uncorrelated".
        """
        img_tuples = []
        for ch_idx in range(len(self._data)):
            if ch_idx == 0:
                # dataset index becomes sample index because all channels have the same
                # length
                img_tuples.append(self._get_img(0, index))
            else:
                # get a random index from corresponding channel
                sample_index = np.random.randint(
                    self.idx_manager.total_grid_count()[0][ch_idx]
                )
                img_tuples.append(self._get_img(ch_idx, sample_index))
        return img_tuples

    def __getitem__(
        self, index: Union[int, tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:

        # Uncorrelated channels means crops to create the input are taken from different
        # spatial locations of the image.
        if (
            self._uncorrelated_channels
            and np.random.rand() < self._uncorrelated_channel_probab
        ):
            input_tuples = self.get_uncorrelated_img_tuples(index)
        else:
            # 0 is the channel index, because in this case locations are the same for
            # all channels
            # tuple for compatibility with _compute_input. #TODO check
            input_tuples = (self._get_img(0, index),)

        if self._enable_rotation:
            input_tuples = self._rotate(input_tuples)

        # Weight the individual channels, typically alpha is fixed
        inp, alpha = self._compute_input(input_tuples)

        target = self._compute_target(input_tuples, alpha)
        norm_target = self.normalize_target(target)

        return inp, norm_target


class LCMultiChDloaderRef(MultiChDloaderRef):
    def __init__(
        self,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
    ):
        self._padding_kwargs = (
            data_config.padding_kwargs  # mode=padding_mode, constant_values=constant_value
        )
        self._uncorrelated_channel_probab = data_config.uncorrelated_channel_probab

        super().__init__(
            data_config,
            fpath,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

        if data_config.overlapping_padding_kwargs is not None:
            assert (
                self._padding_kwargs == data_config.overlapping_padding_kwargs
            ), "During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None"

        else:
            self._overlapping_padding_kwargs = data_config.padding_kwargs

        self.multiscale_lowres_count = data_config.multiscale_lowres_count
        assert self.multiscale_lowres_count is not None
        self._scaled_data = [self._data]
        self._scaled_noise_data = [self._noise_data]

        assert (
            isinstance(self.multiscale_lowres_count, int)
            and self.multiscale_lowres_count >= 1
        )
        assert isinstance(self._padding_kwargs, dict)
        assert "mode" in self._padding_kwargs

        for _ in range(1, self.multiscale_lowres_count):
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

    def reduce_data(
        self, t_list=None, h_start=None, h_end=None, w_start=None, w_end=None
    ):
        assert t_list is not None
        assert h_start is None
        assert h_end is None
        assert w_start is None
        assert w_end is None

        self._data = self._data[t_list].copy()
        self._scaled_data = [
            self._scaled_data[i][t_list].copy() for i in range(len(self._scaled_data))
        ]

        if self._noise_data is not None:
            self._noise_data = self._noise_data[t_list].copy()
            self._scaled_noise_data = [
                self._scaled_noise_data[i][t_list].copy()
                for i in range(len(self._scaled_noise_data))
            ]

        self.N = len(t_list)
        # TODO where tf is self._img_sz defined?
        self.set_img_sz([self._img_sz, self._img_sz], self._grid_sz)
        print(
            f"[{self.__class__.__name__}] Data reduced. New data shape: {self._data.shape}"
        )

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f" Pad:{self._padding_kwargs}"
        if self._uncorrelated_channels:
            msg += f" UncorrChProbab:{self._uncorrelated_channel_probab}"
        return msg

    def _load_scaled_img(
        self, scaled_index, index: Union[int, tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index

        # tidx = self.idx_manager.get_t(idx)
        patch_loc_list = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        nidx = patch_loc_list[0]

        imgs = self._scaled_data[scaled_index][nidx]
        imgs = tuple([imgs[None, ..., i] for i in range(imgs.shape[-1])])
        if self._noise_data is not None:
            noisedata = self._scaled_noise_data[scaled_index][nidx]
            noise = tuple([noisedata[None, ..., i] for i in range(noisedata.shape[-1])])
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            imgs = tuple([img + noise[0] * factor for img in imgs])
        return imgs

    def _crop_img(self, img: np.ndarray, patch_start_loc: tuple):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        max_len_vals = list(self.idx_manager.data_shape[1:-1])
        max_len_vals[-2:] = img.shape[-2:]
        return self._crop_img_with_padding(
            img, patch_start_loc, max_len_vals=max_len_vals
        )

    def _get_img(self, index: int):
        """
        Returns the primary patch along with low resolution patches centered on the primary patch.
        """
        # Noise_tuples is populated when there is synthetic noise in training
        # Should have similar type of noise with the noise model
        # Starting with microsplit, dump the noise, use it instead as an augmentation if nessesary
        img_tuples, noise_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            patch_start_loc = self._get_random_hw(h, w)
            if self._3Ddata:
                patch_start_loc = (
                    np.random.choice(img_tuples[0].shape[-3] - self._depth3D),
                ) + patch_start_loc
        else:
            patch_start_loc = self._get_deterministic_loc(index)

        # LC logic is located here, the function crops the image of the highest resolution
        cropped_img_tuples = [
            self._crop_flip_img(img, patch_start_loc, False, False)
            for img in img_tuples
        ]
        cropped_noise_tuples = [
            self._crop_flip_img(noise, patch_start_loc, False, False)
            for noise in noise_tuples
        ]
        patch_start_loc = list(patch_start_loc)
        h_start, w_start = patch_start_loc[-2], patch_start_loc[-1]
        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {
            i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))
        }
        for scale_idx in range(1, self.multiscale_lowres_count):
            # Returning the image of the lower resolution
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2

            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2
            patch_start_loc[-2:] = [h_start, w_start]
            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, patch_start_loc, False, False)
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

    def __getitem__(self, index: Union[int, tuple[int, int]]):
        img_tuples, noise_tuples = self._get_img(index)
        if self._uncorrelated_channels:
            assert (
                self._input_idx is None
            ), "Uncorrelated channels is not implemented when there is a separate input channel."
            if np.random.rand() < self._uncorrelated_channel_probab:
                img_tuples_new = [None] * len(img_tuples)
                img_tuples_new[0] = img_tuples[0]
                for i in range(1, len(img_tuples)):
                    new_index = np.random.randint(len(self))
                    img_tuples_tmp, _ = self._get_img(new_index)
                    img_tuples_new[i] = img_tuples_tmp[i]
                img_tuples = img_tuples_new

        if self._is_train:
            if self._empty_patch_replacement_enabled:
                if np.random.rand() < self._empty_patch_replacement_probab:
                    img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            img_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # add noise to input, if noise is present combine it with the image
        # factor is for the compute input not to have too much noise because the average of two gaussians
        if len(noise_tuples) > 0:
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            input_tuples = []
            for x in img_tuples:
                x = (
                    x.copy()
                )  # to avoid changing the original image since it is later used for target
                # NOTE: other LC levels already have noise added. So, we just need to add noise to the highest resolution.
                x[0] = x[0] + noise_tuples[0] * factor
                input_tuples.append(x)
        else:
            input_tuples = img_tuples

        # Compute the input by sum / average the channels
        # Alpha is an amount of weight which is applied to the channels when combining them
        # How to sample alpha is still under research
        inp, alpha = self._compute_input(input_tuples)
        target_tuples = [img[:1] for img in img_tuples]
        # add noise to target.
        if len(noise_tuples) >= 1:
            target_tuples = [
                x + noise for x, noise in zip(target_tuples, noise_tuples[1:])
            ]

        target = self._compute_target(target_tuples, alpha)

        norm_target = self.normalize_target(target)

        output = [inp, norm_target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)

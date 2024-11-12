import numpy as np


class IndexSwitcher:
    """
    The idea is to switch from valid indices for target to invalid indices for target.
    If index in invalid for the target, then we return all zero vector as target.
    This combines both logic:
    1. Using less amount of total data.
    2. Using less amount of target data but using full data.
    """

    def __init__(self, idx_manager, data_config, patch_size) -> None:
        self.idx_manager = idx_manager
        self._data_shape = self.idx_manager.get_data_shape()
        self._training_validtarget_fraction = data_config.get(
            "training_validtarget_fraction", 1.0
        )
        self._validtarget_ceilT = int(
            np.ceil(self._data_shape[0] * self._training_validtarget_fraction)
        )
        self._patch_size = patch_size
        assert (
            data_config.deterministic_grid is True
        ), "This only works when the dataset has deterministic grid. Needed randomness comes from this class."
        assert (
            "grid_size" in data_config and data_config.grid_size == 1
        ), "We need a one to one mapping between index and h, w, t"

        self._h_validmax, self._w_validmax = self.get_reduced_frame_size(
            self._data_shape[:3], self._training_validtarget_fraction
        )
        if self._h_validmax < self._patch_size or self._w_validmax < self._patch_size:
            print(
                "WARNING: The valid target size is smaller than the patch size. This will result in all zero target. so, we are ignoring this frame for target."
            )
            self._h_validmax = 0
            self._w_validmax = 0

        print(
            f"[{self.__class__.__name__}] Target Indices: [0,{self._validtarget_ceilT - 1}]. Index={self._validtarget_ceilT - 1} has shape [:{self._h_validmax},:{self._w_validmax}].  Available data: {self._data_shape[0]}"
        )

    def get_valid_target_index(self):
        """
        Returns an index which corresponds to a frame which is expected to have a target.
        """
        _, h, w, _ = self._data_shape
        framepixelcount = h * w
        targetpixels = np.array(
            [framepixelcount] * (self._validtarget_ceilT - 1)
            + [self._h_validmax * self._w_validmax]
        )
        targetpixels = targetpixels / np.sum(targetpixels)
        t = np.random.choice(self._validtarget_ceilT, p=targetpixels)
        # t = np.random.randint(0, self._validtarget_ceilT) if self._validtarget_ceilT >= 1 else 0
        h, w = self.get_valid_target_hw(t)
        index = self.idx_manager.idx_from_hwt(h, w, t)
        # print('Valid', index, h,w,t)
        return index

    def get_invalid_target_index(self):
        # if self._validtarget_ceilT == 0:
        # TODO: There may not be enough data for this to work. The better way is to skip using 0 for invalid target.
        # t = np.random.randint(1, self._data_shape[0])
        # elif self._validtarget_ceilT < self._data_shape[0]:
        #     t = np.random.randint(self._validtarget_ceilT, self._data_shape[0])
        # else:
        #     t = self._validtarget_ceilT - 1
        # 5
        # 1.2 => 2
        total_t, h, w, _ = self._data_shape
        framepixelcount = h * w
        available_h = h - self._h_validmax
        if available_h < self._patch_size:
            available_h = 0
        available_w = w - self._w_validmax
        if available_w < self._patch_size:
            available_w = 0

        targetpixels = np.array(
            [available_h * available_w]
            + [framepixelcount] * (total_t - self._validtarget_ceilT)
        )
        t_probab = targetpixels / np.sum(targetpixels)
        t = np.random.choice(
            np.arange(self._validtarget_ceilT - 1, total_t), p=t_probab
        )

        h, w = self.get_invalid_target_hw(t)
        index = self.idx_manager.idx_from_hwt(h, w, t)
        # print('Invalid', index, h,w,t)
        return index

    def get_valid_target_hw(self, t):
        """
        This is the opposite of get_invalid_target_hw. It returns a h,w which is valid for target.
        This is only valid for single frame setup.
        """
        if t == self._validtarget_ceilT - 1:
            h = np.random.randint(0, self._h_validmax - self._patch_size)
            w = np.random.randint(0, self._w_validmax - self._patch_size)
        else:
            h = np.random.randint(0, self._data_shape[1] - self._patch_size)
            w = np.random.randint(0, self._data_shape[2] - self._patch_size)
        return h, w

    def get_invalid_target_hw(self, t):
        """
        This is the opposite of get_valid_target_hw. It returns a h,w which is not valid for target.
        This is only valid for single frame setup.
        """
        if t == self._validtarget_ceilT - 1:
            h = np.random.randint(
                self._h_validmax, self._data_shape[1] - self._patch_size
            )
            w = np.random.randint(
                self._w_validmax, self._data_shape[2] - self._patch_size
            )
        else:
            h = np.random.randint(0, self._data_shape[1] - self._patch_size)
            w = np.random.randint(0, self._data_shape[2] - self._patch_size)
        return h, w

    def _get_tidx(self, index):
        if isinstance(index, int) or isinstance(index, np.int64):
            idx = index
        else:
            idx = index[0]
        return self.idx_manager.get_t(idx)

    def index_should_have_target(self, index):
        tidx = self._get_tidx(index)
        if tidx < self._validtarget_ceilT - 1:
            return True
        elif tidx > self._validtarget_ceilT - 1:
            return False
        else:
            h, w, _ = self.idx_manager.hwt_from_idx(index)
            return (
                h + self._patch_size < self._h_validmax
                and w + self._patch_size < self._w_validmax
            )

    @staticmethod
    def get_reduced_frame_size(data_shape_nhw, fraction):
        n, h, w = data_shape_nhw

        framepixelcount = h * w
        targetpixelcount = int(n * framepixelcount * fraction)

        # We are currently supporting this only when there is just one frame.
        # if np.ceil(pixelcount / framepixelcount) > 1:
        #     return None, None

        lastframepixelcount = targetpixelcount % framepixelcount
        assert data_shape_nhw[1] == data_shape_nhw[2]
        if lastframepixelcount > 0:
            new_size = int(np.sqrt(lastframepixelcount))
            return new_size, new_size
        else:
            assert (
                targetpixelcount / framepixelcount >= 1
            ), "This is not possible in euclidean space :D (so this is a bug)"
            return h, w

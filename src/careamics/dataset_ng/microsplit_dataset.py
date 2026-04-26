"""MicroSplit dataset: CareamicsDataset subclass for LVAE channel-separation models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.dataset_ng.dataset import ImageRegionData, _adjust_shape_for_channels
from careamics.dataset_ng.image_stack import InMemoryImageStack
from careamics.dataset_ng.normalization.mean_std_normalization import MeanStdNormalization
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patch_extractor.patch_construction import (
    lateral_context_patch_constr,
)
from careamics.dataset_ng.patching_strategies import (
    FixedPatchingStrategy,
    PatchSpecs,
    PatchingStrategy,
    RandomPatchingStrategy,
)
from careamics.transforms import Compose


class MicroSplitDataset(Dataset):
    """Dataset for MicroSplit (LVAE) channel-separation training.

    This dataset implements the MicroSplit input synthesis pipeline:

    1. Multi-channel fluorescence images are loaded into a single image stack.
    2. A multi-scale lateral-context (LC) patch is extracted for each channel.
    3. Channels are combined via alpha-weighted superposition to create the
       synthetic input (mimicking a multi-channel microscope observation).
    4. Target is the full-resolution patch per channel.
    5. Input and target are normalized separately (one-mu-std for input,
       per-channel for target), matching the legacy ``LCMultiChDloader``.

    This class intentionally does **not** call ``CareamicsDataset.__init__``:
    the normalization and patch-extraction pipelines differ fundamentally from
    the standard single-channel workflow, so a composition approach is used
    while still inheriting the ``ImageRegionData`` type contract.

    Parameters
    ----------
    data_config : NGDataConfig
        Base dataset configuration (mode, axes, augmentations, etc.).
    train_data : numpy.ndarray
        Array of shape ``(S, C, Y, X)`` in SC(Z)YX order, where S is the
        number of samples and C is the number of channels.
    multiscale_count : int
        Number of LC scales to produce, including the full-resolution level.
    padding_mode : {"reflect", "wrap"}
        Boundary padding mode for the LC patch constructor.
    input_is_sum : bool
        If ``True``, multiply the alpha-averaged input by ``C`` to get the sum.
    mix_uncorrelated_channels : bool
        If ``True``, channels 1…C-1 are independently sampled from random
        locations on each ``__getitem__`` call with probability
        ``uncorrelated_channel_probab``.
    uncorrelated_channel_probab : float
        Probability of applying the uncorrelated-channel swap. Only relevant
        when ``mix_uncorrelated_channels=True``.
    alpha_range : tuple of (float, float) or None
        Uniform sampling range ``(start, end)`` for per-channel alpha weights.
        ``None`` → equal weights ``1 / C``.
    patching_strategy : PatchingStrategy or None, optional
        If provided, overrides the patching strategy derived from the config.
        Useful for testing with a ``FixedPatchingStrategy``.
    seed : int or None, optional
        Random seed for the internal RNG (uncorrelated sampling, alpha). When
        ``None`` the config seed is used if available.
    """

    def __init__(
        self,
        data_config: NGDataConfig,
        train_data: NDArray,
        multiscale_count: int = 1,
        padding_mode: Literal["reflect", "wrap"] = "reflect",
        input_is_sum: bool = False,
        mix_uncorrelated_channels: bool = False,
        uncorrelated_channel_probab: float = 0.5,
        alpha_range: tuple[float, float] | None = None,
        patching_strategy: PatchingStrategy | None = None,
        seed: int | None = None,
    ) -> None:
        """Constructor — see class docstring for parameter descriptions."""
        super().__init__()

        self.config = data_config

        # ── Image stack ───────────────────────────────────────────────────────
        # train_data is expected in SC(Z)YX order (e.g. SCYX for 2-D).
        image_stack = InMemoryImageStack(
            source="array",
            data=train_data,
            original_axes=data_config.axes,
            original_data_shape=train_data.shape,
        )
        self._image_stack = image_stack

        # ── Patch extractor with lateral-context constructor ──────────────────
        lc_constructor = lateral_context_patch_constr(multiscale_count, padding_mode)
        self.input_extractor: PatchExtractor[InMemoryImageStack] = PatchExtractor(
            [image_stack], patch_constructor=lc_constructor
        )

        # ── Patching strategy ─────────────────────────────────────────────────
        if patching_strategy is not None:
            self.patching_strategy: PatchingStrategy = patching_strategy
        else:
            self.patching_strategy = RandomPatchingStrategy(
                data_shapes=[image_stack.data_shape],
                patch_size=list(data_config.patching.patch_size),
                seed=seed if seed is not None else getattr(data_config, "seed", None),
            )

        # ── Normalization statistics ──────────────────────────────────────────
        # Compute directly from the raw array (mirrors compute_mean_std in
        # the legacy MultiChDloader with target_separate_normalization=True,
        # use_one_mu_std=True, input_is_sum=False).
        raw = train_data.astype(np.float64)  # use float64 for stat precision
        n_channels = train_data.shape[1]

        # Input: global mean/std across all samples, all channels, all pixels.
        global_mean = float(raw.mean())
        global_std = float(raw.std())

        # Target: per-channel mean/std (channel is axis 1 in SCYX).
        per_ch_means = [float(raw[:, c, ...].mean()) for c in range(n_channels)]
        per_ch_stds = [float(raw[:, c, ...].std()) for c in range(n_channels)]

        # For the input (L, H, W) array, `n_channels = L` in MeanStdNormalization.
        # Using a length-1 list means the single stat is broadcast across all L.
        self._norm_input = MeanStdNormalization(
            input_means=[global_mean],
            input_stds=[global_std],
        )
        # Target normalization is per-channel (length == C, one stat per channel).
        self._norm_target = MeanStdNormalization(
            input_means=per_ch_means,
            input_stds=per_ch_stds,
        )

        # ── LC / mixing parameters ────────────────────────────────────────────
        self.multiscale_count = multiscale_count
        self.padding_mode = padding_mode
        self.input_is_sum = input_is_sum
        self.mix_uncorrelated_channels = mix_uncorrelated_channels
        self.uncorrelated_channel_probab = uncorrelated_channel_probab
        self.alpha_range = alpha_range
        self._n_channels = n_channels

        # Internal RNG (used for uncorrelated sampling and alpha draws).
        _seed = seed if seed is not None else getattr(data_config, "seed", None)
        self._rng = np.random.default_rng(seed=_seed)

        # ── Augmentations ─────────────────────────────────────────────────────
        # Reuse the Compose pipeline from the config (training mode only).
        from careamics.config.data.ng_data_config import Mode

        if data_config.mode == Mode.TRAINING:
            self.transforms: Compose | None = Compose(
                list(data_config.augmentations)
            )
        else:
            self.transforms = Compose([])

    # ── Protocol ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of patches in the dataset.

        Returns
        -------
        int
            Equal to ``patching_strategy.n_patches``.
        """
        return self.patching_strategy.n_patches

    def __getitem__(
        self, index: int
    ) -> tuple[ImageRegionData, ImageRegionData]:
        """Return a ``(input_region, target_region)`` pair.

        Parameters
        ----------
        index : int
            Dataset index; routed through the patching strategy.

        Returns
        -------
        tuple of ImageRegionData
            ``(input_region, target_region)`` where:
            - ``input_region.data`` has shape ``(L, (Z), Y, X)``
            - ``target_region.data`` has shape ``(C, (Z), Y, X)``
        """
        patch_spec = self.patching_strategy.get_patch_spec(index)

        if (
            self.mix_uncorrelated_channels
            and self._rng.random() < self.uncorrelated_channel_probab
        ):
            patches, patch_specs = self._get_uncorrelated_patches(index, patch_spec)
            primary_spec = patch_specs[0]
        else:
            patches = self._extract_lc_patch(patch_spec)
            primary_spec = patch_spec

        # patches shape: (C, L, (Z), Y, X)
        # Alpha superposition along C → input (L, (Z), Y, X)
        alphas = self._sample_alphas()
        n_extra_dims = len(patches.shape) - 1  # all dims except C
        alpha_bc = np.array(alphas)[
            :, *(np.newaxis for _ in range(n_extra_dims))
        ]
        input_patch = (alpha_bc * patches).sum(axis=0).astype(np.float32)

        if self.input_is_sum:
            input_patch = input_patch * self._n_channels

        # Target: first LC level (full-res) for each channel → (C, (Z), Y, X)
        target_patch = patches[:, 0, ...].astype(np.float32)

        # Normalization (applies in-place style via callable).
        input_patch, _ = self._norm_input(input_patch)
        target_patch, _ = self._norm_target(target_patch)

        # Augmentation (operates on C(Z)YX arrays; skip for input since it's L(Z)YX).
        # TODO: apply augmentation to input/target jointly when transforms expect C(Z)YX.

        # Wrap into ImageRegionData.
        input_region = self._make_image_region(input_patch, primary_spec)
        target_region = self._make_image_region(target_patch, primary_spec)

        return input_region, target_region

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _extract_lc_patch(self, patch_spec: PatchSpecs) -> NDArray:
        """Extract an ``(C, L, (Z), Y, X)`` LC patch for all channels.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification returned by the patching strategy.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(C, L, (Z), Y, X)``.
        """
        return self.input_extractor.extract_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )

    def _get_uncorrelated_patches(
        self, idx: int, primary_spec: PatchSpecs
    ) -> tuple[NDArray, list[PatchSpecs]]:
        """Sample patches where each channel uses an independent spatial location.

        Channel 0 always uses ``idx``; channels 1…C-1 use random indices.

        Parameters
        ----------
        idx : int
            Dataset index for channel 0.
        primary_spec : PatchSpecs
            Pre-computed patch spec for channel 0.

        Returns
        -------
        patches : numpy.ndarray
            Shape ``(C, L, (Z), Y, X)``.
        patch_specs : list of PatchSpecs
            One spec per channel.
        """
        n_patches = self.patching_strategy.n_patches
        random_indices = self._rng.integers(n_patches, size=(self._n_channels - 1,))

        specs: list[PatchSpecs] = [primary_spec]
        for rand_idx in random_indices:
            specs.append(self.patching_strategy.get_patch_spec(int(rand_idx)))

        channel_patches = []
        for spec in specs:
            ch_patch = self.input_extractor.extract_patch(
                data_idx=spec["data_idx"],
                sample_idx=spec["sample_idx"],
                coords=spec["coords"],
                patch_size=spec["patch_size"],
            )
            # ch_patch: (C, L, (Z), Y, X) — take one channel
            channel_patches.append(ch_patch[len(channel_patches) : len(channel_patches) + 1])

        return np.concatenate(channel_patches, axis=0), specs

    def _sample_alphas(self) -> list[float]:
        """Sample per-channel alpha weights.

        Returns
        -------
        list of float
            Length ``n_channels``.
        """
        if self.alpha_range is None:
            return [1.0 / self._n_channels] * self._n_channels
        start, end = self.alpha_range
        return [float(self._rng.uniform(start, end)) for _ in range(self._n_channels)]

    def _make_image_region(
        self, data: NDArray, patch_spec: PatchSpecs
    ) -> ImageRegionData:
        """Wrap a patch array and its spec into an ``ImageRegionData``.

        Parameters
        ----------
        data : numpy.ndarray
            Patch data in ``(L, (Z), Y, X)`` or ``(C, (Z), Y, X)`` format.
        patch_spec : PatchSpecs
            Patch specification (used for metadata).

        Returns
        -------
        ImageRegionData
            Named tuple with all metadata fields populated.
        """
        image_stack = self._image_stack
        data_shape = _adjust_shape_for_channels(
            shape=image_stack.data_shape,
            channels=self.config.channels,
        )
        return ImageRegionData(
            data=data,
            source=str(image_stack.source),
            data_shape=data_shape,
            dtype=str(image_stack.data_dtype),
            axes=self.config.axes,
            original_data_shape=image_stack.original_data_shape,
            region_spec=patch_spec,
            additional_metadata={},
        )

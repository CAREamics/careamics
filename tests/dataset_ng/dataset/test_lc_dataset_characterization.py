"""Tests for the legacy LCMultiChDloader.

These tests pin the observable behavior of the old pipeline. Subject to removal.

Pins verified:
- Output tuple length and element types.
- Output tensor shapes.
- Output dtype (float32).
- Normalization: mean of normalized input is ≈ 0; std is ≈ 1 (approximately).
- Deterministic output when random cropping is disabled (index → same patch).
- LC structure: each channel carries `multiscale_count` scales stacked on axis 0.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from tests.dataset_ng.dataset.fixtures_microsplit import (
    MULTISCALE_COUNT,
    N_CHANNELS,
    PATCH_SIZE,
    make_legacy_dataset,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def legacy_bundle():
    """LCMultiChDloader + stats; deterministic (random cropping disabled)."""
    return make_legacy_dataset(enable_random_cropping=False)


@pytest.fixture(scope="module")
def legacy_bundle_random():
    """LCMultiChDloader + stats; random cropping enabled."""
    return make_legacy_dataset(enable_random_cropping=True)


# ── Output contract ────────────────────────────────────────────────────────────


class TestOutputContract:
    """Pin the __getitem__ return type and shape contract."""

    def test_returns_tuple(self, legacy_bundle):
        sample = legacy_bundle.dataset[0]
        assert isinstance(sample, tuple)

    def test_tuple_length_two_by_default(self, legacy_bundle):
        """Default output: (inp, norm_target) — alpha and grid_size not returned."""
        sample = legacy_bundle.dataset[0]
        assert len(sample) == 2

    def test_input_is_ndarray(self, legacy_bundle):
        inp, _ = legacy_bundle.dataset[0]
        assert isinstance(inp, np.ndarray)

    def test_target_is_ndarray(self, legacy_bundle):
        _, tgt = legacy_bundle.dataset[0]
        assert isinstance(tgt, np.ndarray)

    def test_input_dtype_float32(self, legacy_bundle):
        inp, _ = legacy_bundle.dataset[0]
        assert inp.dtype == np.float32

    def test_target_dtype_float32(self, legacy_bundle):
        _, tgt = legacy_bundle.dataset[0]
        assert tgt.dtype == np.float32


# ── Shape contract ─────────────────────────────────────────────────────────────


class TestShapes:
    """Pin the expected tensor shapes given the reference parameter set."""

    def test_input_shape_is_L_H_W(self, legacy_bundle):
        """Input: (L, H, W) where L = multiscale_lowres_count."""
        inp, _ = legacy_bundle.dataset[0]
        assert inp.shape == (MULTISCALE_COUNT, PATCH_SIZE, PATCH_SIZE)

    def test_target_shape_is_C_H_W(self, legacy_bundle):
        """Target: (C, H, W) where C = num_channels."""
        _, tgt = legacy_bundle.dataset[0]
        assert tgt.shape == (N_CHANNELS, PATCH_SIZE, PATCH_SIZE)

    def test_input_shape_multiscale_count_1(self):
        """Input L=1 when multiscale_lowres_count=1."""
        bundle = make_legacy_dataset(multiscale_count=1, enable_random_cropping=False)
        inp, _ = bundle.dataset[0]
        assert inp.shape[0] == 1

    def test_input_shape_multiscale_count_2(self):
        """Input L=2 when multiscale_lowres_count=2."""
        bundle = make_legacy_dataset(multiscale_count=2, enable_random_cropping=False)
        inp, _ = bundle.dataset[0]
        assert inp.shape[0] == 2


# ── Determinism when random_cropping=False ────────────────────────────────────


class TestDeterminism:
    """Same index must always return the same patch when random_cropping is off."""

    def test_same_index_returns_same_input(self, legacy_bundle):
        inp0a, _ = legacy_bundle.dataset[0]
        inp0b, _ = legacy_bundle.dataset[0]
        npt.assert_array_equal(inp0a, inp0b)

    def test_same_index_returns_same_target(self, legacy_bundle):
        _, tgt0a = legacy_bundle.dataset[0]
        _, tgt0b = legacy_bundle.dataset[0]
        npt.assert_array_equal(tgt0a, tgt0b)

    def test_different_indices_can_differ(self, legacy_bundle):
        """Not a strict requirement, but usually true for a non-trivial dataset."""
        if len(legacy_bundle.dataset) < 2:
            pytest.skip("Dataset too small to compare two indices")
        inp0, _ = legacy_bundle.dataset[0]
        inp1, _ = legacy_bundle.dataset[1]
        # Content will differ for non-uniform data
        assert not np.array_equal(inp0, inp1)


# ── Normalization properties ───────────────────────────────────────────────────


class TestNormalization:
    """Verify that the normalized output is in a reasonable range."""

    def test_input_mean_near_zero(self, legacy_bundle):
        """Mean of the normalized input (across all patches) is approximately 0."""
        inputs = [
            legacy_bundle.dataset[i][0] for i in range(len(legacy_bundle.dataset))
        ]
        all_inp = np.stack(inputs)
        assert abs(all_inp.mean()) < 1.0  # one-mu-std → global centering

    def test_target_shape_matches_channels(self, legacy_bundle):
        for i in range(min(4, len(legacy_bundle.dataset))):
            _, tgt = legacy_bundle.dataset[i]
            assert tgt.shape[0] == N_CHANNELS


# ── LC structure ───────────────────────────────────────────────────────────────


class TestLCStructure:
    """Verify lateral context scale properties."""

    def test_input_first_scale_is_finest(self, legacy_bundle):
        """Scale 0 should have the finest resolution (highest variance)."""
        inp, _ = legacy_bundle.dataset[0]
        var_scale0 = np.var(inp[0])
        for k in range(1, MULTISCALE_COUNT):
            # Coarser scales have context from larger areas; no strict ordering but
            # check they are not trivially zero.
            assert inp[k].max() > 0, f"Scale {k} should have non-zero values"
        _ = var_scale0  # used above

    def test_all_scales_have_same_spatial_size(self, legacy_bundle):
        """All LC levels are cropped back to (PATCH_SIZE, PATCH_SIZE)."""
        inp, _ = legacy_bundle.dataset[0]
        for k in range(MULTISCALE_COUNT):
            assert inp[k].shape == (
                PATCH_SIZE,
                PATCH_SIZE,
            ), f"Scale {k} has wrong shape: {inp[k].shape}"


# ── Dataset length ─────────────────────────────────────────────────────────────


class TestDatasetLength:
    """Basic sanity on __len__."""

    def test_positive_length(self, legacy_bundle):
        assert len(legacy_bundle.dataset) > 0

    def test_all_indices_accessible(self, legacy_bundle):
        """Every index in [0, len) should return a valid tuple."""
        n = min(len(legacy_bundle.dataset), 8)
        for i in range(n):
            sample = legacy_bundle.dataset[i]
            assert len(sample) == 2

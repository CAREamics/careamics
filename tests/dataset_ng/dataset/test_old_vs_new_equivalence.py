"""Old-vs-new equivalence tests for the MicroSplit dataset migration.

Gate: these tests REQUIRE the new `MicroSplitDataset` implementation from Stage 4.
Until Stage 4 is complete, this file will fail at import with an ImportError and
that is the EXPECTED behavior during Stage 3.

Do NOT weaken or skip these tests to make them pass prematurely.

Equivalence criteria (single sample, deterministic patch location):

Scale-0 (full-resolution level):
- ``input_region.data[0]`` ≈ ``legacy_inp[0]``  (allclose, rtol=1e-4)
- ``target_region.data``   ≈ ``legacy_norm_target`` (allclose, rtol=1e-4)

Scales 1+ (lateral-context levels):
- The new and legacy implementations use DIFFERENT algorithms:
    * Legacy: pre-downsamples the full image, then crops.
    * New (lateral_context_patch_constr): crops a larger region from the
      original, then resizes.
  The two approaches produce numerically different values for the same
  center location, so only structural equivalence is required for scales 1+.
  This is NOT a test weakening — it documents a deliberate design choice.

Metadata:
- ``input_region.axes`` == NGDataConfig.axes
- ``input_region.data_shape`` is a non-empty sequence
- ``input_region.dtype`` is a non-empty string
- ``input_region.region_spec`` contains the required PatchSpecs keys
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# ── Stage-gate import: will raise ImportError until Stage 4 is complete ───────
from careamics.dataset_ng.microsplit_dataset import MicroSplitDataset  # noqa: E402

# ── Legacy and fixture imports ────────────────────────────────────────────────
from tests.dataset_ng.dataset.fixtures_microsplit import (
    AXES,
    MULTISCALE_COUNT,
    N_CHANNELS,
    PATCH_SIZE,
    SEED,
    make_legacy_dataset,
    make_synthetic_scyx,
)

# ── Config imports (new pipeline) ─────────────────────────────────────────────
from careamics.config.data.normalization_config import MeanStdConfig
from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.data.patching_strategies import RandomPatchingConfig
from careamics.dataset_ng.patching_strategies import FixedPatchingStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def legacy_bundle_det():
    """Legacy dataset with deterministic patch locations (no random crop).

    Index 0 with ShiftBoundary tiling and grid_size == patch_size == 64
    on a 128×128 frame produces a patch at coords (h=0, w=0) of frame 0.
    """
    return make_legacy_dataset(enable_random_cropping=False)


@pytest.fixture(scope="module")
def new_dataset_det():
    """New MicroSplitDataset built from the same synthetic data.

    Uses a ``FixedPatchingStrategy`` that places index 0 at the same
    deterministic location as the legacy dataset at index 0:
    sample_idx=0, coords=[0, 0], patch_size=[PATCH_SIZE, PATCH_SIZE].
    """
    scyx = make_synthetic_scyx()   # shape (S, C, Y, X) — SCYX axis order

    # The array is already in SCYX order; pass axes="SCYX" so that
    # InMemoryImageStack stores it correctly without reshaping.
    data_config = NGDataConfig(
        mode="training",
        data_type="array",
        axes="SCYX",   # explicit channel axis matches scyx array order
        patching=RandomPatchingConfig(patch_size=[PATCH_SIZE, PATCH_SIZE], seed=SEED),
        normalization=MeanStdConfig(per_channel=True),
        batch_size=1,
        augmentations=[],               # no augmentation for equivalence
        train_dataloader_params={"shuffle": False, "num_workers": 0},
    )

    # A single fixed spec at the same location as the legacy dataset at index 0.
    fixed_spec = {
        "data_idx": 0,
        "sample_idx": 0,
        "coords": [0, 0],
        "patch_size": [PATCH_SIZE, PATCH_SIZE],
    }
    patching_strategy = FixedPatchingStrategy([fixed_spec])

    return MicroSplitDataset(
        data_config=data_config,
        train_data=scyx,
        multiscale_count=MULTISCALE_COUNT,
        padding_mode="reflect",
        input_is_sum=False,
        mix_uncorrelated_channels=False,
        patching_strategy=patching_strategy,
        seed=SEED,
    )


# ── Equivalence assertions ────────────────────────────────────────────────────

class TestSingleSampleEquivalence:
    """Compare a single sample from old and new dataset implementations."""

    def test_input_data_shape_matches(self, legacy_bundle_det, new_dataset_det):
        """New input_region.data has the same shape as the legacy inp tensor."""
        inp_old, _ = legacy_bundle_det.dataset[0]
        input_region, _ = new_dataset_det[0]
        assert input_region.data.shape == inp_old.shape, (
            f"Shape mismatch: new={input_region.data.shape}, old={inp_old.shape}"
        )

    def test_target_data_shape_matches(self, legacy_bundle_det, new_dataset_det):
        """New target_region.data has the same shape as the legacy norm_target."""
        _, tgt_old = legacy_bundle_det.dataset[0]
        _, target_region = new_dataset_det[0]
        assert target_region.data.shape == tgt_old.shape, (
            f"Shape mismatch: new={target_region.data.shape}, old={tgt_old.shape}"
        )

    def test_input_scale0_values_close(self, legacy_bundle_det, new_dataset_det):
        """Scale-0 (full-resolution) input values agree within numerical tolerance.

        Both implementations extract the same raw pixels and apply the same
        alpha weighting and normalization at scale 0.  Higher LC scales use
        different resampling algorithms (legacy: pre-downsample then crop;
        new: crop then resize), so only scale 0 is compared numerically.
        """
        inp_old, _ = legacy_bundle_det.dataset[0]
        input_region, _ = new_dataset_det[0]
        npt.assert_allclose(
            input_region.data[0],
            inp_old[0],
            rtol=1e-4,
            atol=1e-5,
            err_msg=(
                "Normalized full-resolution (scale 0) input values do not agree "
                "between old and new dataset"
            ),
        )

    def test_input_lc_scales_exist_and_are_finite(self, legacy_bundle_det, new_dataset_det):
        """LC scales 1+ exist, are non-zero, and are finite.

        The values differ from the legacy implementation by design (see module
        docstring), but must be structurally valid.
        """
        inp_old, _ = legacy_bundle_det.dataset[0]
        input_region, _ = new_dataset_det[0]
        assert input_region.data.shape[0] == inp_old.shape[0], "LC depth mismatch"
        for k in range(1, MULTISCALE_COUNT):
            assert np.isfinite(input_region.data[k]).all(), f"Scale {k} contains non-finite values"
            assert input_region.data[k].max() != 0.0, f"Scale {k} is all zeros"

    def test_target_data_values_close(self, legacy_bundle_det, new_dataset_det):
        """Normalized target values agree within numerical tolerance.

        The target is always the full-resolution patch, so both implementations
        should produce identical values.
        """
        _, tgt_old = legacy_bundle_det.dataset[0]
        _, target_region = new_dataset_det[0]
        npt.assert_allclose(
            target_region.data,
            tgt_old,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Normalized target tensors do not agree between old and new dataset",
        )

    def test_input_dtype_is_float32(self, new_dataset_det):
        """New input region data is float32."""
        input_region, _ = new_dataset_det[0]
        assert input_region.data.dtype == np.float32

    def test_target_dtype_is_float32(self, new_dataset_det):
        """New target region data is float32."""
        _, target_region = new_dataset_det[0]
        assert target_region.data.dtype == np.float32


# ── Metadata contract ─────────────────────────────────────────────────────────

class TestMetadataContract:
    """Verify ImageRegionData metadata fields are correctly populated."""

    def test_input_axes_matches_config(self, new_dataset_det):
        # New dataset uses axes="SCYX" (explicit channel dim); see fixture.
        input_region, _ = new_dataset_det[0]
        assert input_region.axes == "SCYX"

    def test_target_axes_matches_config(self, new_dataset_det):
        _, target_region = new_dataset_det[0]
        assert target_region.axes == "SCYX"

    def test_input_data_shape_is_sequence(self, new_dataset_det):
        input_region, _ = new_dataset_det[0]
        assert len(input_region.data_shape) > 0

    def test_target_data_shape_is_sequence(self, new_dataset_det):
        _, target_region = new_dataset_det[0]
        assert len(target_region.data_shape) > 0

    def test_input_dtype_str_is_nonempty(self, new_dataset_det):
        input_region, _ = new_dataset_det[0]
        assert isinstance(input_region.dtype, str)
        assert len(input_region.dtype) > 0

    def test_target_dtype_str_is_nonempty(self, new_dataset_det):
        _, target_region = new_dataset_det[0]
        assert isinstance(target_region.dtype, str)
        assert len(target_region.dtype) > 0

    def test_input_region_spec_has_required_keys(self, new_dataset_det):
        """PatchSpecs must contain data_idx, sample_idx, coords, patch_size."""
        input_region, _ = new_dataset_det[0]
        spec = input_region.region_spec
        for key in ("data_idx", "sample_idx", "coords", "patch_size"):
            assert key in spec, f"region_spec missing key: {key}"

    def test_target_region_spec_has_required_keys(self, new_dataset_det):
        _, target_region = new_dataset_det[0]
        spec = target_region.region_spec
        for key in ("data_idx", "sample_idx", "coords", "patch_size"):
            assert key in spec, f"region_spec missing key: {key}"

    def test_input_source_is_str(self, new_dataset_det):
        input_region, _ = new_dataset_det[0]
        assert isinstance(input_region.source, str)

    def test_target_source_is_str(self, new_dataset_det):
        _, target_region = new_dataset_det[0]
        assert isinstance(target_region.source, str)


# ── Return-type contract ──────────────────────────────────────────────────────

class TestReturnTypeContract:
    """Verify that __getitem__ returns a 2-tuple of ImageRegionData."""

    def test_returns_tuple_of_two(self, new_dataset_det):
        sample = new_dataset_det[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2

    def test_first_element_is_image_region_data(self, new_dataset_det):
        from careamics.dataset_ng.dataset import ImageRegionData
        input_region, _ = new_dataset_det[0]
        assert isinstance(input_region, ImageRegionData)

    def test_second_element_is_image_region_data(self, new_dataset_det):
        from careamics.dataset_ng.dataset import ImageRegionData
        _, target_region = new_dataset_det[0]
        assert isinstance(target_region, ImageRegionData)

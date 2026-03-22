"""Tests for the N2VManipulateTorch fast path (fast=True).

These tests verify:
1. Shape equivalence between reference (fast=False) and fast (fast=True) paths.
2. Masking semantics: mask == (manipulated != original) for both paths.
3. Reproducibility: same seed + same fast flag → same outputs.
4. Struct mask bounds: replaced pixels stay in [batch_min, batch_max].
5. _apply_struct_mask_torch_vec functional equivalence with _apply_struct_mask_torch.
6. Smoke training loop: N steps with fast=True produces valid outputs.
"""

import pytest
import torch

from careamics.config.augmentations import N2VManipulateConfig
from careamics.config.support import SupportedPixelManipulation
from careamics.transforms import N2VManipulateTorch
from careamics.transforms.pixel_manipulation_torch import (
    _apply_struct_mask_torch,
    _apply_struct_mask_torch_vec,
    _get_stratified_coords_torch,
)
from careamics.transforms.struct_mask_parameters import StructMaskParameters

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    strategy: str = "uniform",
    struct_axis: str = "none",
    seed: int = SEED,
) -> N2VManipulateConfig:
    return N2VManipulateConfig(
        roi_size=5,
        masked_pixel_percentage=5.0,
        strategy=strategy,
        struct_mask_axis=struct_axis,
        struct_mask_span=3,
        seed=seed,
    )


def _make_batch(batch: int, channels: int, *spatial: int) -> torch.Tensor:
    """Return a float tensor with shape (B, C, *spatial) filled with arange values."""
    total = batch * channels
    for s in spatial:
        total *= s
    return torch.arange(total, dtype=torch.float32).reshape(batch, channels, *spatial)


# ---------------------------------------------------------------------------
# 1. Shape equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "median"])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 32, 32),  # 2D, single channel
        (2, 3, 32, 32),  # 2D, multi-channel
        (1, 1, 8, 16, 16),  # 3D, single channel
    ],
)
def test_fast_output_shapes_match_reference(strategy: str, shape: tuple) -> None:
    """Fast path must return tensors with the same shapes as the reference."""
    batch = _make_batch(*shape)

    ref = N2VManipulateTorch(_make_config(strategy=strategy), device="cpu", fast=False)
    fast = N2VManipulateTorch(_make_config(strategy=strategy), device="cpu", fast=True)

    m_ref, o_ref, mk_ref = ref(batch)
    m_fast, o_fast, mk_fast = fast(batch)

    assert m_fast.shape == m_ref.shape
    assert o_fast.shape == o_ref.shape
    assert mk_fast.shape == mk_ref.shape


# ---------------------------------------------------------------------------
# 2. Masking semantics: mask == (manipulated != original)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "median"])
@pytest.mark.parametrize("use_fast", [False, True])
def test_mask_equals_diff(strategy: str, use_fast: bool) -> None:
    """mask must equal the boolean diff (manipulated != original) for both paths."""
    batch = _make_batch(2, 2, 32, 32)
    aug = N2VManipulateTorch(
        _make_config(strategy=strategy), device="cpu", fast=use_fast
    )
    manipulated, original, mask = aug(batch)

    diff = (manipulated != original).to(torch.uint8)
    assert torch.equal(diff, mask), (
        f"mask != diff for strategy={strategy}, fast={use_fast}"
    )


@pytest.mark.parametrize("use_fast", [False, True])
def test_mask_subset_of_diff_with_struct(use_fast: bool) -> None:
    """With struct masking, mask tracks center pixels only; diff is a superset.

    The struct pass writes additional neighboring pixels not captured in mask.
    This is the defined reference behavior: mask positions ⊆ diff positions.
    """
    batch = _make_batch(2, 1, 32, 32)
    aug = N2VManipulateTorch(
        _make_config(strategy="uniform", struct_axis="horizontal"),
        device="cpu",
        fast=use_fast,
    )
    manipulated, original, mask = aug(batch)
    diff = (manipulated != original).to(torch.uint8)

    # Every pixel flagged in mask must also appear in diff
    assert (mask & ~diff).sum() == 0, "mask contains positions not in diff"
    # The struct pass adds extra changes, so diff >= mask in terms of set pixels
    assert diff.sum() >= mask.sum()


# ---------------------------------------------------------------------------
# 3. Reproducibility: same seed → same outputs (within the same path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "median"])
@pytest.mark.parametrize("use_fast", [False, True])
def test_reproducibility_same_seed(strategy: str, use_fast: bool) -> None:
    """Two instances with the same seed and same fast flag produce identical outputs."""
    batch = _make_batch(2, 1, 32, 32)
    cfg = _make_config(strategy=strategy, seed=SEED)

    aug1 = N2VManipulateTorch(cfg, device="cpu", fast=use_fast)
    aug2 = N2VManipulateTorch(cfg, device="cpu", fast=use_fast)

    m1, o1, mk1 = aug1(batch)
    m2, o2, mk2 = aug2(batch)

    assert torch.equal(m1, m2)
    assert torch.equal(mk1, mk2)


@pytest.mark.parametrize("strategy", ["uniform", "median"])
@pytest.mark.parametrize("use_fast", [False, True])
def test_subsequent_calls_differ(strategy: str, use_fast: bool) -> None:
    """Consecutive calls on the same instance advance the RNG and differ."""
    batch = _make_batch(2, 1, 32, 32)
    aug = N2VManipulateTorch(
        _make_config(strategy=strategy), device="cpu", fast=use_fast
    )
    _, _, mk1 = aug(batch)
    _, _, mk2 = aug(batch)
    assert not torch.equal(mk1, mk2)


# ---------------------------------------------------------------------------
# 4. Struct mask bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_fast", [False, True])
@pytest.mark.parametrize("struct_axis", ["horizontal", "vertical"])
def test_struct_replaced_pixels_in_range(use_fast: bool, struct_axis: str) -> None:
    """Pixels replaced by struct masking must lie in [batch_min, batch_max]."""
    batch = _make_batch(2, 1, 32, 32)
    aug = N2VManipulateTorch(
        _make_config(strategy="uniform", struct_axis=struct_axis),
        device="cpu",
        fast=use_fast,
    )
    manipulated, original, _ = aug(batch)

    # Check only the pixels that changed
    changed = manipulated != original
    if not changed.any():
        pytest.skip("No pixels were changed — increase mask percentage.")

    for b in range(batch.shape[0]):
        bmin = original[b].min().item()
        bmax = original[b].max().item()
        changed_vals = manipulated[b][changed[b]]
        assert changed_vals.min().item() >= bmin - 1e-5
        assert changed_vals.max().item() <= bmax + 1e-5


# ---------------------------------------------------------------------------
# 5. _apply_struct_mask_torch_vec functional equivalence
# ---------------------------------------------------------------------------


def test_vec_struct_mask_output_shape() -> None:
    """Vectorized struct mask must return a patch with the same shape."""
    rng_ref = torch.Generator(device="cpu").manual_seed(SEED)
    rng_vec = torch.Generator(device="cpu").manual_seed(SEED)

    patch = _make_batch(3, 16, 16).squeeze(1)  # shape (3, 16, 16) — batch of 2D
    coords_rng = torch.Generator(device="cpu").manual_seed(SEED)
    coords = _get_stratified_coords_torch(5.0, patch.shape, coords_rng)

    struct_params = StructMaskParameters(axis=0, span=3)

    patch_ref = patch.clone()
    patch_vec = patch.clone()

    out_ref = _apply_struct_mask_torch(patch_ref, coords, struct_params, rng_ref)
    out_vec = _apply_struct_mask_torch_vec(patch_vec, coords, struct_params, rng_vec)

    assert out_ref.shape == out_vec.shape


def test_vec_struct_mask_values_in_range() -> None:
    """Values written by the vectorized variant must be in [batch_min, batch_max]."""
    rng_vec = torch.Generator(device="cpu").manual_seed(SEED)

    patch = _make_batch(3, 16, 16).squeeze(1).float()
    original = patch.clone()

    coords_rng = torch.Generator(device="cpu").manual_seed(SEED)
    coords = _get_stratified_coords_torch(5.0, patch.shape, coords_rng)
    struct_params = StructMaskParameters(axis=0, span=3)

    out = _apply_struct_mask_torch_vec(patch.clone(), coords, struct_params, rng_vec)

    changed = out != original
    if not changed.any():
        pytest.skip("No pixels changed.")

    for b in range(original.shape[0]):
        bmin = original[b].min().item()
        bmax = original[b].max().item()
        changed_vals = out[b][changed[b]]
        assert changed_vals.min().item() >= bmin - 1e-5
        assert changed_vals.max().item() <= bmax + 1e-5


def test_vec_struct_mask_same_positions_changed() -> None:
    """Both variants must write to the same pixel positions."""
    rng_ref = torch.Generator(device="cpu").manual_seed(SEED)
    rng_vec = torch.Generator(device="cpu").manual_seed(SEED + 1)

    patch = _make_batch(3, 16, 16).squeeze(1).float()
    coords_rng = torch.Generator(device="cpu").manual_seed(SEED)
    coords = _get_stratified_coords_torch(5.0, patch.shape, coords_rng)
    struct_params = StructMaskParameters(axis=0, span=3)
    original = patch.clone()

    out_ref = _apply_struct_mask_torch(patch.clone(), coords, struct_params, rng_ref)
    out_vec = _apply_struct_mask_torch_vec(
        patch.clone(), coords, struct_params, rng_vec
    )

    changed_ref = (out_ref != original).nonzero(as_tuple=False)
    changed_vec = (out_vec != original).nonzero(as_tuple=False)

    assert torch.equal(changed_ref, changed_vec), (
        "Reference and vectorized variants must modify the same pixel positions."
    )


# ---------------------------------------------------------------------------
# 6. Smoke training loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "median"])
def test_smoke_training_loop_fast(strategy: str) -> None:
    """Simulate N training steps with fast=True; outputs must be valid tensors."""
    cfg = _make_config(strategy=strategy, seed=SEED)
    aug = N2VManipulateTorch(cfg, device="cpu", fast=True)

    n_steps = 10
    for _ in range(n_steps):
        batch = torch.randn(2, 1, 32, 32)
        manipulated, original, mask = aug(batch)

        assert manipulated.shape == batch.shape
        assert original.shape == batch.shape
        assert mask.shape == batch.shape

        assert not torch.isnan(manipulated).any(), "NaN in manipulated output"
        assert not torch.isinf(manipulated).any(), "Inf in manipulated output"
        assert mask.sum() > 0, "Empty mask — no pixels were masked"
        assert mask.dtype == torch.uint8


@pytest.mark.parametrize("strategy", ["uniform", "median"])
def test_smoke_training_loop_fast_3d(strategy: str) -> None:
    """Simulate training steps on 3D batches with fast=True."""
    cfg = _make_config(strategy=strategy, seed=SEED)
    aug = N2VManipulateTorch(cfg, device="cpu", fast=True)

    for _ in range(5):
        batch = torch.randn(1, 1, 8, 16, 16)
        manipulated, original, mask = aug(batch)

        assert manipulated.shape == batch.shape
        assert not torch.isnan(manipulated).any()
        assert mask.sum() > 0


def test_smoke_training_loop_fast_with_struct() -> None:
    """Smoke test with struct mask enabled and fast=True."""
    cfg = _make_config(strategy="uniform", struct_axis="vertical", seed=SEED)
    aug = N2VManipulateTorch(cfg, device="cpu", fast=True)

    for _ in range(5):
        batch = torch.randn(2, 2, 32, 32)
        manipulated, original, mask = aug(batch)

        assert manipulated.shape == batch.shape
        assert not torch.isnan(manipulated).any()
        assert mask.sum() > 0

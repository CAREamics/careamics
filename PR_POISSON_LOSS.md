## Description

> [!NOTE]
> **tldr**: This PR introduces Poisson negative log-likelihood (NLL) loss for N2V, enabling proper handling of count data with Poisson noise, including automatic denormalisation and multi-channel support.

### Background - why do we need this PR?

Standard N2V loss uses mean squared error (MSE), which assumes Gaussian noise. For count data, where noise follows a Poisson distribution, MSE would be suboptimal, because Poisson noise variance equals mean, whilst MSE assumes constant variance.

Examples of Poisson count data include photon counting (fluorescence microscopy), particle counting, or any discrete counting process. This PR adds `n2v_poisson` loss to support Poisson noise models whilst maintaining backwards compatibility with the existing MSE-based N2V loss.

### Overview - what changed?

This PR adds a new Poisson NLL loss function for N2V and enhances the existing N2V loss to handle channel mismatches. Configuration support has been added to allow users to select between MSE-based (`n2v`) and Poisson NLL-based (`n2v_poisson`) losses. The implementation includes automatic denormalisation from normalised training data back to count scale before computing the Poisson NLL.

### Implementation - how did you implement the changes?

The `n2v_poisson_loss` function leverages PyTorch's optimised `F.poisson_nll_loss` and implements:
- **Automatic denormalisation**: Predictions and targets are denormalised back to count scale using provided image_means/image_stds (because Poisson NLL requires non-negative counts, but normalised training data can be negative)
- **ReLU activation + epsilon**: Ensures predictions are positive (Poisson λ > 0 requirement) whilst allowing near-zero values
- **Channel mismatch handling**: Identically to `n2v_loss` for consistency
- **Masked averaging**: Computes mean loss over masked pixels only (standard N2V pattern)

Usage:
```python
config = N2VAlgorithm(
    algorithm="n2v",
    loss="n2v_poisson",  # Use Poisson NLL for count data
    ...
)
```

The `n2v_loss` function was also enhanced to handle channel mismatches (e.g., model outputs 1 channel but input has multiple channels), which is useful for certain multi-channel denoising scenarios.

## Changes Made

### New features or files
- `n2v_poisson_loss()` function in `src/careamics/losses/fcn/losses.py`
- `tests/losses/test_fcn_losses.py` - comprehensive tests for Poisson loss and channel handling
- `N2V_POISSON` enum in `src/careamics/config/support/supported_losses.py`
- `SOFTPLUS` enum in `src/careamics/config/support/supported_activations.py`

### Modified features or files
- `n2v_loss()` in `src/careamics/losses/fcn/losses.py` - handles channel mismatches
- `loss_factory()` in `src/careamics/losses/loss_factory.py` - added Poisson loss
- `__all__` in `src/careamics/losses/__init__.py` - exported `n2v_poisson_loss`
- `loss` field in `src/careamics/config/algorithms/n2v_algorithm_config.py` - now `Literal["n2v", "n2v_poisson"]`
- `get_activation()` in `src/careamics/models/activation.py` - added Softplus support

### Removed features or files
- None

## How has this been tested?

**61 tests passing** in `tests/losses/test_fcn_losses.py` covering:
- Various batch sizes (1, 4) and image sizes (32, 64)
- Single and multi-channel inputs (1, 2 channels)
- Channel mismatches (model outputs 1 channel, input has 2-3 channels)
- Edge cases (zero predictions, empty masks)
- Loss factory integration for both `"n2v"` and `"n2v_poisson"` string/enum types

## Related Issues

- None

## Breaking changes

None - Existing N2V configurations with `loss="n2v"` continue to work exactly as before.

## Additional Notes and Examples

**Why ReLU over Softplus for Poisson?**
- ReLU has no floor (outputs can be arbitrarily close to 0)
- Softplus has a floor of ~0.693, which is problematic for sparse/low-count data
- The epsilon (1e-6) prevents log(0) whilst allowing near-zero predictions

**Denormalisation is needed**
- Poisson NLL is only meaningful in count space (non-negative values)
- Training data is typically normalised (mean/std), which can produce negative values
- The loss automatically denormalises using provided `image_means`/`image_stds` from `*args`

---

**Please ensure your PR meets the following requirements:**

- [x] Code builds and passes tests locally, including doctests
- [x] New tests have been added (for bug fixes/features)
- [ ] Pre-commit passes
- [ ] PR to the documentation exists (for bug fixes / features)

"""Test denormalization with per-channel N2V models."""

import numpy as np
import pytest

from careamics.transforms import Denormalize


def test_denormalize_without_target_indices():
    """Test denormalization without target indices (old behavior)."""
    # 4-channel statistics
    means = [0.2, 0.5, 0.7, 1.0]
    stds = [0.6, 0.8, 1.1, 1.5]

    # Create normalized prediction (1 output channel, batch size 2)
    # Simulating normalized output (mean≈0, std≈1)
    prediction = np.random.randn(2, 1, 64, 64).astype(np.float32)

    # Denormalize without target indices (uses first channel stats)
    denorm = Denormalize(image_means=means, image_stds=stds)
    output = denorm(prediction)

    # Should use channel 0 stats: mean=0.2, std=0.6
    expected_mean = 0.2
    expected_std = 0.6 + 1e-6

    # Check output is approximately denormalized with channel 0 stats
    # Allow some tolerance due to random input
    assert output.shape == prediction.shape
    assert abs(output.mean() - expected_mean) < 0.5  # Rough check
    assert abs(output.std() - expected_std) < 0.5


def test_denormalize_with_target_indices_single_channel():
    """Test denormalization with target indices for single channel."""
    # 4-channel statistics
    means = [0.2, 0.5, 0.7, 1.0]
    stds = [0.6, 0.8, 1.1, 1.5]

    # Create normalized prediction (1 output channel, batch size 2)
    # Use fixed values for easier testing
    prediction = np.zeros((2, 1, 64, 64), dtype=np.float32)

    # Test with channel 2
    denorm = Denormalize(
        image_means=means, image_stds=stds, target_channel_indices=[2]
    )
    output = denorm(prediction)

    # Input is all zeros (mean=0 in normalized space)
    # After denormalization: output = 0 * (std + eps) + mean = mean
    expected_output = means[2]  # Should use channel 2 mean

    assert output.shape == prediction.shape
    assert np.allclose(output, expected_output, atol=1e-5)


def test_denormalize_with_target_indices_multiple_channels():
    """Test denormalization with target indices for multiple channels."""
    # 4-channel statistics
    means = [0.2, 0.5, 0.7, 1.0]
    stds = [0.6, 0.8, 1.1, 1.5]

    # Create normalized prediction (2 output channels, batch size 2)
    prediction = np.zeros((2, 2, 64, 64), dtype=np.float32)

    # Test with channels 1 and 3
    denorm = Denormalize(
        image_means=means, image_stds=stds, target_channel_indices=[1, 3]
    )
    output = denorm(prediction)

    # Check channel 0 uses stats from input channel 1
    assert np.allclose(output[:, 0, :, :], means[1], atol=1e-5)

    # Check channel 1 uses stats from input channel 3
    assert np.allclose(output[:, 1, :, :], means[3], atol=1e-5)


def test_denormalize_channel_2_bug():
    """Test the specific bug: channel 2 model getting channel 0 stats."""
    # Real JUMP dataset statistics
    means = [0.19541956, 0.45363748, 0.6717809, 0.92672324]
    stds = [0.60383224, 0.8025205, 1.0712336, 1.4702153]

    # Simulated normalized prediction from channel 2 model
    # In normalized space: mean≈0, std≈1
    np.random.seed(42)
    prediction_normalized = np.random.randn(1, 1, 128, 128).astype(np.float32)

    # OLD BEHAVIOR (WRONG): Uses channel 0 stats
    denorm_old = Denormalize(image_means=means, image_stds=stds)
    output_old = denorm_old(prediction_normalized)

    # Expected old behavior: denormalized with channel 0 stats
    expected_old_mean = means[0]

    # NEW BEHAVIOR (CORRECT): Uses channel 2 stats
    denorm_new = Denormalize(
        image_means=means, image_stds=stds, target_channel_indices=[2]
    )
    output_new = denorm_new(prediction_normalized)

    # Expected new behavior: denormalized with channel 2 stats
    expected_new_mean = means[2]

    # Verify outputs are different
    assert not np.allclose(output_old, output_new)

    # Verify old output is around channel 0 mean
    assert abs(output_old.mean() - expected_old_mean) < abs(
        output_old.mean() - expected_new_mean
    )

    # Verify new output is around channel 2 mean
    assert abs(output_new.mean() - expected_new_mean) < abs(
        output_new.mean() - expected_old_mean
    )


def test_denormalize_target_indices_length_mismatch():
    """Test error when target indices length doesn't match output channels."""
    means = [0.2, 0.5, 0.7, 1.0]
    stds = [0.6, 0.8, 1.1, 1.5]

    prediction = np.zeros((2, 1, 64, 64), dtype=np.float32)

    # Provide 2 target indices for 1 output channel
    denorm = Denormalize(
        image_means=means, image_stds=stds, target_channel_indices=[1, 2]
    )

    with pytest.raises(ValueError, match="does not match number of output channels"):
        denorm(prediction)


def test_denormalize_actual_values():
    """Test denormalization with actual values to ensure correctness."""
    means = [1.0, 2.0, 3.0]
    stds = [0.5, 1.0, 1.5]
    eps = 1e-6

    # Normalized input: value of 2.0 in normalized space
    normalized_value = 2.0
    prediction = np.full((1, 1, 10, 10), normalized_value, dtype=np.float32)

    # Without target indices (uses channel 0)
    denorm = Denormalize(image_means=means, image_stds=stds, target_channel_indices=[0])
    output = denorm(prediction)

    # Expected: normalized_value * (std[0] + eps) + mean[0]
    expected = normalized_value * (stds[0] + eps) + means[0]

    assert np.allclose(output, expected, atol=1e-4)

    # With target indices (uses channel 2)
    denorm = Denormalize(image_means=means, image_stds=stds, target_channel_indices=[2])
    output = denorm(prediction)

    # Expected: normalized_value * (std[2] + eps) + mean[2]
    expected = normalized_value * (stds[2] + eps) + means[2]

    assert np.allclose(output, expected, atol=1e-4)

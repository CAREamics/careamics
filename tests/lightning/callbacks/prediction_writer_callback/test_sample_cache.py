"""Test the utility `SampleCache` class used by the WriteTile classes."""

import numpy as np
import pytest

from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    caches,
)


def test_add(samples):
    # n_samples_per_file = [3, 1, 2]
    # simulated batch size = 1
    samples, n_samples_per_file = samples
    sample_cache = caches.SampleCache(n_samples_per_file)
    sample_cache.add(samples[0])
    np.testing.assert_array_equal(sample_cache.sample_cache[0], samples[0])


def test_has_all_file_samples_true(samples):
    # n_samples_per_file = [3, 1, 2]
    # simulated batch size = 1
    samples, n_samples_per_file = samples
    sample_cache = caches.SampleCache(n_samples_per_file)
    for i in range(n_samples_per_file[0]):
        sample_cache.add(samples[i])
    assert sample_cache.has_all_file_samples()


def test_has_all_file_samples_false(samples):
    # n_samples_per_file = [3, 1, 2]
    # simulated batch size = 1
    samples, n_samples_per_file = samples
    sample_cache = caches.SampleCache(n_samples_per_file)
    for i in range(n_samples_per_file[0] - 1):
        sample_cache.add(samples[i])
    assert not sample_cache.has_all_file_samples()


def test_has_all_file_samples_error(samples):
    # n_samples_per_file = [3, 1, 2]
    # simulated batch size = 1
    samples, n_samples_per_file = samples
    sample_cache = caches.SampleCache(n_samples_per_file)
    # simulate iterating through all files
    for sample in samples:
        sample_cache.add(sample)
    for _ in n_samples_per_file:
        sample_cache.pop_file_samples()

    with pytest.raises(ValueError):
        sample_cache.has_all_file_samples()


def test_pop_file_samples(samples):
    # n_samples_per_file = [3, 1, 2]
    # simulated batch size = 1
    samples, n_samples_per_file = samples
    sample_cache = caches.SampleCache(n_samples_per_file)

    n = 4
    for i in range(n):
        sample_cache.add(samples[i])

    first_file_n_samples = n_samples_per_file[0]

    # assert popped samples correct
    popped_samples = sample_cache.pop_file_samples()
    for i in range(first_file_n_samples):
        np.testing.assert_array_equal(popped_samples[i], samples[i])

    # assert remaining sample correct
    np.testing.assert_array_equal(
        sample_cache.sample_cache[0], samples[first_file_n_samples]
    )


def test_pop_file_samples_error(samples):
    # n_samples_per_file = [3, 1, 2]
    # simulated batch size = 1
    samples, n_samples_per_file = samples
    sample_cache = caches.SampleCache(n_samples_per_file)
    sample_cache.add(samples[0])

    # should raise an error because sample cache does not contain all file samples
    with pytest.raises(ValueError):
        sample_cache.pop_file_samples()


def test_reset(samples):
    # n_samples_per_file = [3, 1, 2]
    # simulated batch size = 1
    samples, n_samples_per_file = samples
    sample_cache = caches.SampleCache(n_samples_per_file)
    sample_cache.add(samples[0])

    sample_cache.reset()
    assert len(sample_cache.sample_cache) == 0
    assert next(sample_cache.n_samples_iter) == n_samples_per_file[0]

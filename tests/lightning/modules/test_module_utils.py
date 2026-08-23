import torch

from careamics.lightning.modules.module_utils import mmse_and_sample_std


def test_mmse_and_sample_std_matches_the_drawn_samples() -> None:
    """The mean and standard deviation are those of the drawn samples."""
    samples = [torch.full((2, 3, 4, 4), float(i)) for i in range(4)]
    it = iter(samples)

    mean, std = mmse_and_sample_std(lambda _: next(it), torch.empty(0), n_samples=4)

    expected = torch.stack(samples, dim=0)
    torch.testing.assert_close(mean, expected.mean(dim=0))
    torch.testing.assert_close(std, expected.std(dim=0))


def test_mmse_and_sample_std_draws_n_samples() -> None:
    """`sample_prediction` is called exactly `n_samples` times."""
    calls = 0

    def sample(x_data: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.randn(1, 1, 8, 8)

    mmse_and_sample_std(sample, torch.empty(0), n_samples=7)
    assert calls == 7


def test_mmse_and_sample_std_undefined_for_a_single_sample() -> None:
    """With one sample the standard deviation is undefined and reported as None."""
    sample = torch.randn(2, 3, 4, 4)

    mean, std = mmse_and_sample_std(lambda _: sample, torch.empty(0), n_samples=1)

    torch.testing.assert_close(mean, sample)
    assert std is None


def test_mmse_and_sample_std_is_exact_under_affine_sampling() -> None:
    """Sampling in denormalized space scales the std without offsetting it."""
    samples = [torch.randn(2, 1, 4, 4) for _ in range(5)]
    data_mean, data_std = 12.0, 3.0

    it = iter(samples)
    _, normalized_std = mmse_and_sample_std(
        lambda _: next(it), torch.empty(0), n_samples=5
    )

    it = iter(samples)
    _, denormalized_std = mmse_and_sample_std(
        lambda _: next(it) * data_std + data_mean, torch.empty(0), n_samples=5
    )

    torch.testing.assert_close(denormalized_std, normalized_std * data_std)

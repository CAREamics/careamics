import numpy as np
import pytest

from careamics.config import create_ng_data_configuration
from careamics.config.data import MicroSplitDataConfig
from careamics.dataset.factory import MicroSplitMultiplexedTargetData, create_dataset
from careamics.dataset.factory.microsplit_factory import create_microsplit_dataset
from careamics.lightning.data import GroupedIndexSampler


def _assert_indices_grouped(sampler: GroupedIndexSampler):
    """Assert the indices from the sampler are grouped as expected."""

    grouped_indices = sampler.grouped_indices
    sizes = [len(group) for group in grouped_indices]

    # counting that adjacent indices belong to the same index group
    current_group: int | None = None
    prev_group: int | None = None
    group_count: int = 0
    for idx in sampler:
        current_group = next(
            i for i, group in enumerate(grouped_indices) if idx in group
        )
        if prev_group is not None:
            # Switching to the next group should only happen once the all the previous
            # group's indices have been counted
            if current_group != prev_group:
                assert group_count == sizes[prev_group]
                group_count = 0  # reset the count for the new index group
        group_count += 1
        prev_group = current_group


def test_iter():
    """Test the indices from the sampler a grouped as expected."""

    sizes = [4, 21, 1, 8]
    index_bins = np.cumsum(sizes)
    grouped_indices = [
        list(range(index_bins[i - 1], ub)) if i != 0 else list(range(ub))
        for i, ub in enumerate(index_bins)
    ]

    rng = np.random.default_rng(42)
    sampler = GroupedIndexSampler(grouped_indices, rng=rng)

    _assert_indices_grouped(sampler)


def test_from_dataset():
    """
    Test that the indices are grouped as expected when the sampler is created from the
    `classmethod` `from_dataset`.
    """

    rng = np.random.default_rng(42)

    patch_size = (16, 16)
    data_shapes = [(64, 48), (55, 54), (71, 65)]
    input_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_shapes]
    train_data_config = create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=[],
        in_memory=True,
        seed=42,
    )

    train_dataset = create_dataset(
        config=train_data_config,
        inputs=input_data,
        targets=None,
    )

    sampler = GroupedIndexSampler.from_dataset(train_dataset, rng=rng)

    _assert_indices_grouped(sampler)


@pytest.mark.parametrize(
    ("uncorrelated_channel_prob", "expect_error"),
    [(0.0, False), (1.0, True)],
)
def test_from_microsplit_dataset(
    uncorrelated_channel_prob: float,
    expect_error: bool,
) -> None:
    """Test grouped sampling follows MicroSplit constructor index support."""
    config = MicroSplitDataConfig(
        mode="training",
        data_type="array",
        axes="SCYX",
        patching={"name": "stratified", "patch_size": (16, 16), "seed": 42},
        normalization={"name": "none"},
        uncorrelated_channel_prob=uncorrelated_channel_prob,
    )
    data = np.zeros((1, 2, 32, 32), dtype=np.float32)
    dataset = create_microsplit_dataset(
        config=config,
        data=MicroSplitMultiplexedTargetData([data]),
        rng=np.random.default_rng(42),
    )

    if expect_error:
        with pytest.raises(NotImplementedError):
            GroupedIndexSampler.from_dataset(dataset, rng=np.random.default_rng(42))
    else:
        sampler = GroupedIndexSampler.from_dataset(
            dataset, rng=np.random.default_rng(42)
        )
        assert sorted(sampler.grouped_indices[0]) == list(range(len(dataset)))

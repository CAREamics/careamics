import numpy as np

from careamics.config.configuration_factories import _create_ng_data_configuration
from careamics.dataset_ng.dataset import Mode
from careamics.dataset_ng.factory import create_array_dataset
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler


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
    train_data_config = _create_ng_data_configuration(
        data_type="array",
        axes="YX",
        patch_size=patch_size,
        batch_size=1,
        augmentations=[],
        seed=42,
    )

    train_dataset = create_array_dataset(
        config=train_data_config, mode=Mode.TRAINING, inputs=input_data, targets=None
    )

    sampler = GroupedIndexSampler.from_dataset(train_dataset, rng=rng)

    _assert_indices_grouped(sampler)

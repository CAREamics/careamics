from torch.utils.data import Sampler

from careamics.config.data.ng_data_model import NGDataConfig
from careamics.dataset_ng.dataset import CareamicsDataset
from careamics.dataset_ng.patch_filter import create_coord_filter, create_patch_filter

from .filtering_sampler import FilteringSampler


def create_filtering_sampler(
    dataset: CareamicsDataset,
    config: NGDataConfig,
) -> Sampler | None:
    if config.patch_filter is None and config.coord_filter is None:
        return None

    patch_filter = None
    if config.patch_filter is not None:
        patch_filter = create_patch_filter(config.patch_filter)

    coord_filter = None
    if config.coord_filter is not None and dataset.mask_extractor is not None:
        coord_filter = create_coord_filter(
            config.coord_filter, mask=dataset.mask_extractor
        )

    return FilteringSampler(
        dataset=dataset,
        patch_filter=patch_filter,
        coord_filter=coord_filter,
        patience=config.patch_filter_patience,
        seed=config.seed,
    )

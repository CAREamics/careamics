from collections.abc import Sequence

from .patching_strategy_protocol import PatchSpecs


class WholeSamplePatchingStrategy:
    # TODO: warn this strategy should only be used with batch size = 1
    #   for the case of multiple image stacks with different dimensions

    # TODO: docs
    def __init__(self, data_shapes: Sequence[Sequence[int]]):
        self.data_shapes = data_shapes

        self.patch_specs: list[PatchSpecs] = self._initialize_patch_specs()

    @property
    def n_patches(self) -> int:
        return len(self.patch_specs)

    def get_patch_spec(self, index: int) -> PatchSpecs:
        return self.patch_specs[index]

    def _initialize_patch_specs(self) -> list[PatchSpecs]:
        patch_specs: list[PatchSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]
            for sample_idx in range(data_shape[0]):
                patch_specs.append(
                    {
                        "data_idx": data_idx,
                        "sample_idx": sample_idx,
                        "coords": tuple(0 for _ in spatial_shape),
                        "patch_size": spatial_shape,
                    }
                )
        return patch_specs

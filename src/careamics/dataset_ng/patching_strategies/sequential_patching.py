from .patching_strategy_types import PatchSpecs


class SequentialPatchingStrategy:

    def get_patch_spec(self, index: int) -> PatchSpecs:
        raise NotImplementedError("Sequential patching is not implemented yet")

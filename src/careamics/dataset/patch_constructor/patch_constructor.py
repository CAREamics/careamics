"""PatchConstructor Protocol."""

from typing import Any, Protocol

from numpy.typing import NDArray

from careamics.dataset.patching import PatchSpecs


class PatchConstructor(Protocol):
    """Module for extracting and constructing inputs and targets."""

    @property
    def n_patches(self) -> int:
        """The number of patches."""
        ...

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any] | None, PatchSpecs]:
        """Construct the patch that will be input into the model.

        Parameters
        ----------
        index : int
            The index of the patch, the index has to be less than `n_patches`.

        Returns
        -------
        input : NDArray[Any]
            The input patch.
        target : NDArray[Any] | None
            The target patch.
        patch_spec : PatchSpecs
            The patch specification.
        """
        ...

    # e.g. for MicroSplit the full input has the lateral context.
    #      we need to remove the lateral context to calculate the normalization stats.
    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        """Get the principle input.

        This is useful for tasks such as the calculation of stats for normalization.

        Parameters
        ----------
        input : NDArray[Any]
            The complete input.

        Returns
        -------
        NDArray[Any]
            The principal input (C(Z)YX).
        """
        ...

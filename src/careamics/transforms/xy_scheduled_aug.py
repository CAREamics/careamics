"""Deterministic dihedral-8 scheduled augmentation for 2D patches."""

import numpy as np
from numpy.typing import NDArray

from careamics.transforms.transform import Transform

# Dihedral group D4 for a 2D square, encoded as (n_rot90, flip_x, flip_y).
# flip_x flips the last axis (-1), flip_y flips the second-to-last axis (-2).
# Rotations use axes (-2, -1) so they operate on the YX plane regardless of
# the number of leading dimensions (C or CZ).
_DIHEDRAL_OPS: list[tuple[int, bool, bool]] = [
    (0, False, False),  # identity
    (1, False, False),  # rot90
    (2, False, False),  # rot180
    (3, False, False),  # rot270
    (0, True, False),   # flip_x
    (0, False, True),   # flip_y
    (1, True, False),   # rot90 + flip_x
    (1, False, True),   # rot90 + flip_y
]


class XYScheduledAugmentation(Transform):
    """Deterministic 2D geometric augmentation using the dihedral-8 group.

    For each sample the applied transform is chosen as::

        op = DIHEDRAL_OPS[(sample_idx + epoch) % n_transforms]

    This guarantees:

    - **Determinism**: same ``(sample_idx, epoch)`` always selects the same op.
    - **Complementary epochs**: over ``n_transforms`` epochs every sample sees every
      op exactly once, so consecutive epochs produce non-redundant views.
    - **Within-epoch coverage**: different sample indices map to different ops within
      a single epoch (modulo ``n_transforms``).

    The transform expects C(Z)YX input arrays.  Target and any additional arrays
    receive the **identical** geometric operation as the input patch.

    Parameters
    ----------
    n_transforms : int
        Number of dihedral ops to cycle through.  Must be between 1 and 8
        (inclusive).  ``8`` uses the full dihedral group; smaller values use the
        first ``n_transforms`` entries in the group table.

    Attributes
    ----------
    n_transforms : int
        Size of the op cycle.
    epoch : int
        Current epoch.  Updated externally via :meth:`set_epoch`.
    sample_idx : int
        Current sample index within the dataset.  Updated externally via
        :meth:`set_sample_idx`.
    """

    def __init__(self, n_transforms: int = 8) -> None:
        """Constructor.

        Parameters
        ----------
        n_transforms : int, optional
            Number of dihedral ops to use (1–8).  Default is 8.
        """
        if not (1 <= n_transforms <= 8):
            raise ValueError(
                f"`n_transforms` must be between 1 and 8, got {n_transforms}."
            )
        self.n_transforms = n_transforms
        self.epoch: int = 0
        self.sample_idx: int = 0

    # ------------------------------------------------------------------
    # External state setters (called by ScheduledAugCallback / dataset)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set the current training epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        """
        self.epoch = epoch

    def set_sample_idx(self, idx: int) -> None:
        """Set the current sample (dataset) index.

        Parameters
        ----------
        idx : int
            Dataset index of the sample being transformed.
        """
        self.sample_idx = idx

    # ------------------------------------------------------------------
    # Transform application
    # ------------------------------------------------------------------

    def __call__(
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        """Apply the scheduled dihedral op to ``patch`` (and optionally ``target``).

        Parameters
        ----------
        patch : np.ndarray
            Input patch in C(Z)YX format.
        target : np.ndarray or None, optional
            Optional target array in C(Z)YX format.
        **additional_arrays : np.ndarray
            Additional arrays transformed identically to ``patch``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray or None, dict[str, np.ndarray]]
            Transformed patch, transformed target (or None), and transformed
            additional arrays dict.
        """
        op = _DIHEDRAL_OPS[(self.sample_idx + self.epoch) % self.n_transforms]

        patch_out = self._apply_op(patch, op)
        target_out = self._apply_op(target, op) if target is not None else None
        additional_out = {
            key: self._apply_op(arr, op) for key, arr in additional_arrays.items()
        }

        return patch_out, target_out, additional_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_op(
        self, arr: NDArray, op: tuple[int, bool, bool]
    ) -> NDArray:
        """Apply a single dihedral op tuple to ``arr``.

        Parameters
        ----------
        arr : np.ndarray
            Array in C(Z)YX format.
        op : tuple[int, bool, bool]
            ``(n_rot90, flip_x, flip_y)`` where ``n_rot90`` is the number of
            counter-clockwise 90° rotations in the YX plane, ``flip_x`` flips the
            last axis, and ``flip_y`` flips the second-to-last axis.

        Returns
        -------
        np.ndarray
            Transformed array (contiguous copy).
        """
        n_rot90, flip_x, flip_y = op
        out = arr

        if n_rot90 != 0:
            out = np.rot90(out, k=n_rot90, axes=(-2, -1))

        if flip_x:
            out = np.flip(out, axis=-1)

        if flip_y:
            out = np.flip(out, axis=-2)

        return np.ascontiguousarray(out)

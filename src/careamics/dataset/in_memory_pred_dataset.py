"""In-memory prediction dataset."""

from __future__ import annotations

from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.transforms import Compose

from ..config import InferenceConfig
from ..config.transformations import NormalizeModel
from .dataset_utils import reshape_array


class InMemoryPredDataset(Dataset):
    """Simple prediction dataset returning images along the sample axis.

    Parameters
    ----------
    prediction_config : InferenceConfig
        Prediction configuration.
    inputs : NDArray
        Input data.
    """

    def __init__(
        self,
        prediction_config: InferenceConfig,
        inputs: NDArray,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        prediction_config : InferenceConfig
            Prediction configuration.
        inputs : NDArray
            Input data.

        Raises
        ------
        ValueError
            If data_path is not a directory.
        """
        self.pred_config = prediction_config
        self.input_array = inputs
        self.axes = self.pred_config.axes
        self.image_means = self.pred_config.image_means
        self.image_stds = self.pred_config.image_stds

        # Reshape data
        self.data = reshape_array(self.input_array, self.axes)

        # get transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeModel(image_means=self.image_means, image_stds=self.image_stds)
            ],
        )

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[NDArray, ...]:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        tuple(numpy.ndarray, ...)
            Transformed patch.
        """
        return self.patch_transform(patch=self.data[index])

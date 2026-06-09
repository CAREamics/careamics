import numpy as np
import pytest
import torch
from bioimageio.spec import InvalidDescr, load_description
from torch import Tensor

from careamics import CAREamist
from careamics.model_io import export_to_bmz, load_pretrained
from careamics.model_io.bmz_io import _export_state_dict, _load_state_dict

pytestmark = pytest.mark.mps_gh_fail


def test_state_dict_io(tmp_path, ordered_array, pre_trained_v2):
    """Test exporting and loading a state dict."""
    # training data
    train_array = ordered_array((32, 32))
    path = tmp_path / "model.pth"

    # instantiate CAREamist
    careamist = CAREamist(checkpoint_path=pre_trained_v2, work_dir=tmp_path)

    # predict (no tiling and no tta)
    predicted_output, _ = careamist.predict(train_array)
    predicted = np.concatenate(predicted_output, axis=0)

    # save model
    _export_state_dict(careamist.model, path)
    assert path.exists()

    # load model
    _load_state_dict(careamist.model, path)

    # predict (no tiling and no tta)
    predicted_loaded, _ = careamist.predict(train_array)
    np.testing.assert_almost_equal(predicted_loaded[0], predicted, decimal=3)


def test_bmz_io(tmp_path, ordered_array, pre_trained_v2):
    """Test exporting and loading to the BMZ."""
    # training data
    train_array = ordered_array((32, 32))

    # instantiate CAREamist
    careamist = CAREamist(checkpoint_path=pre_trained_v2, work_dir=tmp_path)

    # predict (no tiling and no tta)
    predicted_output, _ = careamist.predict(train_array)
    predicted = np.concatenate(predicted_output, axis=0)

    # export to BioImage Model Zoo
    path = tmp_path / "other_folder" / "model.zip"
    export_to_bmz(
        model=careamist.model,
        config=careamist.config,
        path_to_archive=path,
        model_name="TopModel",
        general_description="A model that just walked in.",
        data_description="My data.",
        authors=[{"name": "Amod", "affiliation": "El"}],
        input_array=train_array[np.newaxis, np.newaxis, ...],
        output_array=predicted[np.newaxis, np.newaxis, ...],
        model_version="0.2.0",
    )
    assert path.exists()

    # load description
    description = load_description(path)
    assert not isinstance(description, InvalidDescr)
    assert str(description.version) == "0.2.0"

    # load model
    config, model = load_pretrained(path)
    model.eval()
    careamist.model.eval()
    assert config == careamist.config

    # compare predictions
    with torch.no_grad():
        torch_array = Tensor(train_array[np.newaxis, np.newaxis, ...])
        predicted = careamist.model.forward(torch_array).numpy().squeeze()
        predicted_loaded = model.forward(torch_array).numpy().squeeze()
    np.testing.assert_almost_equal(predicted_loaded, predicted, decimal=3)

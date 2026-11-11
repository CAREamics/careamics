import numpy as np
import pytest
import torch
from bioimageio.spec import InvalidDescr, load_description
from tests.utils import set_seed
from torch import Tensor

from careamics import CAREamist
from careamics.model_io import export_to_bmz, load_pretrained
from careamics.model_io.bmz_io import _export_state_dict, _load_state_dict
from careamics.utils.version import get_careamics_version


@pytest.mark.parametrize("model_path_fixture", ["pre_trained", "lvae_pre_trained"])
def test_state_dict_io(tmp_path, ordered_array, model_path_fixture, request):
    """Test exporting and loading a state dict."""
    # training data
    train_array = ordered_array((32, 32))
    path = tmp_path / "model.pth"

    # get the model path from the fixture
    model_path = request.getfixturevalue(model_path_fixture)

    # instantiate CAREamist
    careamist = CAREamist(checkpoint_path=model_path, work_dir=tmp_path)

    # predict (no tiling and no tta)
    set_seed(13)
    predicted_output, _ = careamist.predict(train_array)
    predicted = np.concatenate(predicted_output, axis=0)

    # save model
    _export_state_dict(careamist.model, path)
    assert path.exists()

    # load model
    _load_state_dict(careamist.model, path)

    # predict (no tiling and no tta)
    set_seed(13)
    predicted_loaded, _ = careamist.predict(train_array)
    np.testing.assert_almost_equal(predicted_loaded[0], predicted, decimal=3)


@pytest.mark.parametrize("model_path_fixture", ["pre_trained", "lvae_pre_trained"])
def test_bmz_io(request, tmp_path, ordered_array, model_path_fixture):
    """Test exporting and loading to the BMZ."""
    # training data
    train_array = ordered_array((64, 64))

    # get the model path from the fixture
    model_path = request.getfixturevalue(model_path_fixture)

    # instantiate CAREamist
    careamist = CAREamist(checkpoint_path=model_path, work_dir=tmp_path)

    # predict (no tiling and no tta)
    set_seed(13)
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
    )
    assert path.exists()

    # load description
    description = load_description(path)
    assert not isinstance(description, InvalidDescr)
    assert str(description.version) == get_careamics_version()

    # load model
    config, model = load_pretrained(path)
    model.eval()
    careamist.model.eval()
    assert config == careamist.config

    # compare predictions
    with torch.no_grad():
        torch_array = Tensor(train_array[np.newaxis, np.newaxis, ...])
        set_seed(13)
        predicted = careamist.model.forward(torch_array)
        set_seed(13)
        predicted_loaded = model.forward(torch_array)
        # lvae output is a tuple, so we take only the first element
        if isinstance(predicted, tuple):
            predicted = predicted[0]
        if isinstance(predicted_loaded, tuple):
            predicted_loaded = predicted_loaded[0]

    predicted = predicted.numpy().squeeze()
    predicted_loaded = predicted_loaded.numpy().squeeze()
    np.testing.assert_almost_equal(predicted_loaded, predicted, decimal=3)

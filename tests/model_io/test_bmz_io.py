import numpy as np
import pytest
import torch
from bioimageio.spec import InvalidDescr, load_description
from tests.utils import set_seed
from torch import Tensor

from careamics import CAREamist
from careamics.config.factories import create_advanced_pn2v_config
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


def _pn2v_careamist(tmp_path) -> CAREamist:
    """Return an untrained PN2V `CAREamist`."""
    config = create_advanced_pn2v_config(
        experiment_name="pn2v_bmz",
        data_type="array",
        axes="YX",
        patch_size=(32, 32),
        batch_size=2,
        n_prior_samples=2,
        model_params={"depth": 2, "num_channels_init": 8},
    )
    return CAREamist(config, work_dir=tmp_path)


def test_export_to_bmz_rejects_pn2v(tmp_path):
    """PN2V cannot be exported: the UNet outputs prior samples, not the MMSE."""
    careamist = _pn2v_careamist(tmp_path)
    array = np.zeros((1, 1, 32, 32), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="PN2V"):
        export_to_bmz(
            model=careamist.model,
            config=careamist.config,
            path_to_archive=tmp_path / "model.zip",
            model_name="PN2V",
            general_description="PN2V.",
            data_description="My data.",
            authors=[{"name": "Amod", "affiliation": "El"}],
            input_array=array,
            output_array=array,
        )
    assert not (tmp_path / "model.zip").exists()


def test_careamist_export_to_bmz_rejects_pn2v_before_predict(tmp_path):
    """`CAREamist.export_to_bmz` fails before it predicts the example output."""
    careamist = _pn2v_careamist(tmp_path)

    with pytest.raises(NotImplementedError, match="PN2V"):
        careamist.export_to_bmz(
            path_to_archive=tmp_path / "model.zip",
            friendly_model_name="PN2V",
            input_array=np.zeros((32, 32), dtype=np.float32),
            authors=[{"name": "Amod", "affiliation": "El"}],
            general_description="PN2V.",
            data_description="My data.",
        )

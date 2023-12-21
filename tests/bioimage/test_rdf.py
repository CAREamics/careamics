from pathlib import Path

import pytest

from careamics.bioimage.rdf import _get_model_doc, get_default_model_specs


@pytest.mark.parametrize("name", ["Noise2Void"])
def test_get_model_doc(name):
    doc = _get_model_doc(name)
    assert Path(doc).exists()


def test_get_model_doc_error():
    with pytest.raises(FileNotFoundError):
        _get_model_doc("NotAModel")


@pytest.mark.parametrize("name", ["Noise2Void"])
@pytest.mark.parametrize("is_3D", [True, False])
def test_default_model_specs(name, is_3D):
    mean = 666.666
    std = 42.420

    if is_3D:
        axes = "zyx"
    else:
        axes = "yx"

    specs = get_default_model_specs(name, mean, std, is_3D=is_3D)
    assert specs["name"] == name
    assert specs["preprocessing"][0][0]["kwargs"]["mean"] == [mean]
    assert specs["preprocessing"][0][0]["kwargs"]["std"] == [std]
    assert specs["preprocessing"][0][0]["kwargs"]["axes"] == axes
    assert specs["postprocessing"][0][0]["kwargs"]["offset"] == [mean]
    assert specs["postprocessing"][0][0]["kwargs"]["gain"] == [std]
    assert specs["postprocessing"][0][0]["kwargs"]["axes"] == axes

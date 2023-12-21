from pathlib import Path

import numpy as np
import pytest

from careamics import Configuration, Engine


def test_generate_rdf_without_mean_std(minimum_config: dict):
    """Test that generating rdf without specifying mean and std
    raises an error."""
    # create configuration and save it to disk
    config = Configuration(**minimum_config)

    # create an engine to export the model
    engine = Engine(config=config)
    with pytest.raises(ValueError):
        engine._generate_rdf()

    # test if error is raised when config is None
    engine.config = None
    with pytest.raises(ValueError):
        engine._generate_rdf()


def test_bioimage_generate_rdf(minimum_config: dict):
    """Test generating rdf using default specs."""
    # create configuration and save it to disk
    mean = 666.666
    std = 42.420
    minimum_config["data"]["mean"] = mean
    minimum_config["data"]["std"] = std
    minimum_config["data"]["axes"] = "YX"
    config = Configuration(**minimum_config)

    # create an engine to export the model
    engine = Engine(config=config)

    # create a monkey patch for the input
    engine._input = np.random.randint(0, 255, minimum_config["training"]["patch_size"])

    # Sample files
    axes = "bcyx"
    test_inputs = Path(minimum_config["working_directory"]) / "test_inputs.npy"
    test_outputs = Path(minimum_config["working_directory"]) / "test_outputs.npy"

    # Export rdf
    rdf = engine._generate_rdf()
    assert rdf["preprocessing"][0][0]["kwargs"]["mean"] == [mean]
    assert rdf["preprocessing"][0][0]["kwargs"]["std"] == [std]
    assert rdf["postprocessing"][0][0]["kwargs"]["offset"] == [mean]
    assert rdf["postprocessing"][0][0]["kwargs"]["gain"] == [std]
    assert rdf["test_inputs"] == [str(test_inputs)]
    assert rdf["test_outputs"] == [str(test_outputs)]
    assert rdf["input_axes"] == [axes]
    assert rdf["output_axes"] == [axes]


def test_bioimage_generate_rdf_with_specs(minimum_config: dict):
    """Test model export to bioimage format by using default specs."""
    # create configuration and save it to disk
    mean = 666.666
    std = 42.420
    minimum_config["data"]["mean"] = mean
    minimum_config["data"]["std"] = std
    minimum_config["data"]["axes"] = "YX"
    config = Configuration(**minimum_config)

    # create an engine to export the model
    engine = Engine(config=config)

    # create a monkey patch for the input
    engine._input = np.random.randint(0, 255, minimum_config["training"]["patch_size"])

    # Test model specs
    model_specs = {"description": "Some description", "license": "to kill"}
    rdf = engine._generate_rdf(model_specs=model_specs)
    assert rdf["description"] == model_specs["description"]
    assert rdf["license"] == model_specs["license"]


@pytest.mark.parametrize(
    "axes, shape",
    [
        ("YX", (64, 128)),
        ("ZYX", (8, 128, 64)),
    ],
)
def test_bioimage_generate_rdf_with_input(
    minimum_config: dict, ordered_array, axes, shape
):
    """Test generating rdf using default specs."""
    # create configuration and save it to disk
    mean = 666.666
    std = 42.420
    minimum_config["algorithm"]["is_3D"] = len(shape) == 3
    minimum_config["training"]["patch_size"] = (
        (64, 64) if len(shape) == 2 else (8, 64, 64)
    )
    minimum_config["data"]["mean"] = mean
    minimum_config["data"]["std"] = std
    minimum_config["data"]["axes"] = axes
    config = Configuration(**minimum_config)

    # create an engine to export the model
    engine = Engine(config=config)

    # create a monkey patch for the input
    monkey_input = np.random.randint(0, 255, minimum_config["training"]["patch_size"])
    engine._input = monkey_input

    # create other input
    other_input = ordered_array(shape=shape)

    # create rdf
    rdf = engine._generate_rdf(input_array=other_input)

    # inspect input/output
    array_in = np.load(rdf["test_inputs"][0])
    assert (array_in.squeeze() == other_input).all()

    array_out = np.load(rdf["test_outputs"][0])
    assert array_out.max() != 0

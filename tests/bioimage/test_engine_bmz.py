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

    # Sample files
    axes = "bcyx"
    test_inputs, test_outputs = engine._get_sample_io_files()

    # Export rdf
    rdf = engine._generate_rdf()
    assert rdf["preprocessing"][0][0]["kwargs"]["mean"] == [mean]
    assert rdf["preprocessing"][0][0]["kwargs"]["std"] == [std]
    assert rdf["postprocessing"][0][0]["kwargs"]["offset"] == [mean]
    assert rdf["postprocessing"][0][0]["kwargs"]["gain"] == [std]
    assert rdf["test_inputs"] == test_inputs
    assert rdf["test_outputs"] == test_outputs
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

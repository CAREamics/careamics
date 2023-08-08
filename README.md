<p align="center">
  <a href="https://careamics.github.io/">
    <img src="https://raw.githubusercontent.com/CAREamics/.github/main/profile/images/banner_careamics.png">
  </a>
</p>

# CAREamics Restoration

[![License](https://img.shields.io/pypi/l/careamics-restoration.svg?color=green)](https://github.com/CAREamics/careamics-restoration/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/careamics-restoration.svg?color=green)](https://pypi.org/project/careamics)
[![Python Version](https://img.shields.io/pypi/pyversions/careamics-restoration.svg?color=green)](https://python.org)
[![CI](https://github.com/CAREamics/careamics-restoration/actions/workflows/ci.yml/badge.svg)](https://github.com/CAREamics/careamics-restoration/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CAREamics/careamics-restoration/branch/main/graph/badge.svg)](https://codecov.io/gh/CAREamics/careamics-restoration)

## Installation

``` bash
pip install careamics
```
For more details on the options please follow the installation [guide](https://careamics.github.io/careamics-restoration/).

## Usage

CAREamics uses the Engine object to construct the pipeline for both training and prediction. First we import the Engine.
```python
from careamics_restoration.engine import Engine
```
The Engine could be initialized in 2 ways:
1. Using the [yaml config](examples/n2v_2D_reference.yml) file

Specify the mandatory parameters in the config file
```yaml
experiment_name: Name of the experiment
working_directory: Path to the working directory, where all the outputs will be stored

algorithm: 
    loss: type of loss function, e.g. n2v for Noise2Void
    model: model architecture, e.g. UNet
    is_3D: True if 3D data, False if 2D data

training:
  num_epochs: Number of training epochs
  patch_size: Size of the patches, List of 2 or 3 elements
  batch_size: Batch size for training

extraction_strategy: Controls how the patches are extracted from the data

data:
    data_format: File extension, e.g. tif
    axes: Defines the shape of the input data
```
Full description of the configuration parameters is in the [documentation](https://careamics.github.io/careamics-restoration/).


```python
engine = Engine(config_path="config.yml")

```
2. Using the path to the pretrained model
It's also possible to initialize the Engine using the model checkpoint, saved during the training or downloaded from the [BioImage Model Zoo](https://bioimage.io/#/).
Checkpoint must contain model_state_dict.
Read more abount saving and loading models in the [documentation](https://careamics.github.io/careamics-restoration/).

Once Engine is initialized, we can start training, providing the relative paths to train and validation data

```python
engine.train(train_path=train_path, val_path=val_path)
```
Training will run for the specified number of epochs and save the model checkpoint in the working directory.

Prediction could be done directly after the training or by loading the pretrained model checkpoint.

```python
predictions = engine.predict(pred_path=predict_path)
```

For more examples please take a look at the [notebooks](examples).


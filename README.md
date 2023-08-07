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
[![Datasets CI](https://github.com/CAREamics/careamics-restoration/actions/workflows/datasets_ci.yml/badge.svg)](https://github.com/CAREamics/careamics-restoration/actions/workflows/datasets_ci.yml)
[![codecov](https://codecov.io/gh/CAREamics/careamics-restoration/branch/main/graph/badge.svg)](https://codecov.io/gh/CAREamics/careamics-restoration)

## Installation

``` bash
pip install careamics[all]
pip install careamics[wandb]
etc...
```

## Usage

Define the main Engine object using either [yaml config](examples/n2v_2D_reference.yml) or path to pretrained model
```python
from careamics_restoration.engine import Engine

engine = Engine(config_path="n2v_2D_SEM.yml")

```
Start training, providing the relative paths to train and validation data

```python
engine.train(train_path=train_path, val_path=val_path)
```

Do the same for prediction

```python
predictions = engine.predict(pred_path=predict_path)
```

For specific examples please follow the [example notebooks](examples).


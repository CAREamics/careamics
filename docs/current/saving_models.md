---
icon: lucide/save
description: Quick start
---

# Saving and exporting models

CAREamics saves model checkpoints automatically during training. These checkpoints can
be reused to resume prediction, or the model can be exported to the
[BioImage Model Zoo](https://bioimage.io/) (BMZ) format to be shared with the community.

## Checkpoints

During training, CAREamics uses a `ModelCheckpoint` callback that periodically saves the
model weights to disk. Checkpoints are stored under the `work_dir` passed to the
`CAREamist` as follows:

```
<work_dir>/checkpoints/<experiment_name>_<run_version>/
├── <experiment_name>_<epoch>_step_<step>_<val_loss>.ckpt
└── <experiment_name>_last.ckpt
```

The intermediate checkpoints correspond to the epochs with the lowest validation loss,
while `<experiment_name>_last.ckpt` is the checkpoint from the final epoch.

### Listing checkpoints

The `get_checkpoints` method returns the available checkpoint paths, sorted by epoch
number, with the last checkpoint (if present) appended at the end.

```python title="Listing checkpoints"
--8<-- "current/saving_models.py:get_checkpoints"
```

### Reloading a checkpoint

A `CAREamist` can be recreated from any checkpoint path, for instance to run prediction
in a new session.

```python title="Reloading from a checkpoint"
--8<-- "current/saving_models.py:reload"
```

See [Predicting with CAREamics](./careamist_predicting.md) for how to select the best or
last checkpoint at prediction time.

## Exporting to the BioImage Model Zoo

The `export_to_bmz` method packages the trained model into a `.zip` archive that can be
uploaded to the BioImage Model Zoo, making it usable from other tools such as Fiji,
Ilastik and deepImageJ.

```python title="Minimal BMZ export"
--8<-- "tutorials/export_to_bmz_noexec.py:export_min"
```

1. The input array must have the same axes as recorded in the configuration.

For a complete walkthrough of every argument, the archive contents, model validation and
loading a model back, see the
[Exporting to the BioImage Model Zoo](../tutorials/export_to_bmz.md) tutorial.

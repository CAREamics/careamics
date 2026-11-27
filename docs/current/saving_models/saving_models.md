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

### Checkpoint record

The `CAREamist` will automatically create a `checkpoint_record.json` file to record the checkpoints that were created during training.

The `get_checkpoints` method reads `checkpoint_record.json` and returns the structured
record as a dictionary. The record contains the best, last, top-k, and suggested
checkpoints for inference, as well as all checkpoints saved periodically, the monitored
metric, and whether greater or smaller values are considered better. It returns `None`
when no checkpoint record exists.

For example, access the top-k checkpoints saved according to the monitored metric with
the `"best_k"` key:

```python title="Listing checkpoints"
--8<-- "current/saving_models/saving_models.py:get_checkpoints"
```

### Reloading a checkpoint

A `CAREamist` can be recreated from any checkpoint path, for instance to run prediction in a new session. 
Note that, if predicting directly after training the CAREamist will automatically load the selected checkpoint for inference.
To predict with a different checkpoint, a new CAREamist instance must be created.

```python title="Reloading from a checkpoint"
--8<-- "current/saving_models/saving_models.py:reload"
```

The checkpoint record can also be read using the utility function
`read_checkpoint_record`. It takes the checkpoint directory, which contains the
checkpoints and `checkpoint_record.json`, as input. It returns `None` if no record is
found.

```python
--8<-- "current/saving_models/saving_models.py:util-load"
```

See [Predicting with CAREamics](../careamist/careamist_predicting.md) for how to select the best or
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
[Exporting to the BioImage Model Zoo](../../tutorials/export_to_bmz.md) tutorial.

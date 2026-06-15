---
icon: lucide/chart-scatter
description: Logging
---

# Logging

You can follow the progress of the training, or inspect it once training is over, using the logging capacities of PyTorch Lightning.

CAREamics always writes a CSV log of training and validation metrics. In addition, it can forward those metrics to [Weights & Biases](https://wandb.ai/) or [TensorBoard](https://www.tensorflow.org/tensorboard) by passing `logger="wandb"` or `logger="tensorboard"` to the advanced configuration factory (see [Choosing a logger](./configuration.md#choosing-a-logger)). The CSV log is *always* written, regardless of which extra logger is enabled. Enabling and configuring those two backends is covered in the [WandB and TensorBoard tutorial](../tutorials/logging_wandb_tensorboard.md).

All training artefacts are written under the `work_dir` passed to the `CAREamist` (the current working directory by default), with the following layout:

```
work_dir/
├── csv_logs/<experiment_name>/version_<n>/metrics.csv
├── tb_logs/lightning_logs/version_<n>/             # if logger="tensorboard"
├── wandb_logs/                                     # if logger="wandb"
└── checkpoints/<experiment_name>_<n>/
```

A new `version_<n>` directory is created each time training is re-run with the same `experiment_name`.

## CSV logger

The CSV logger is enabled by default. After training, the train and validation losses can be read back through `CAREamist.get_losses()` and plotted:

```python title="Plotting losses from the CSV log"
--8<-- "current/careamist_train_logging.py:csv_logger"
```

1. Returns a dict with keys `train_epoch`, `val_epoch`, `train_loss`, and `val_loss`. Train and validation rows are written separately to the CSV, so the two epoch arrays can have different lengths.

The underlying file is `work_dir/csv_logs/<experiment_name>/version_<n>/metrics.csv` and can be opened directly with any CSV tool.

## Weights & Biases and TensorBoard

Beyond the CSV log, CAREamics can forward metrics to Weights & Biases or TensorBoard by passing `logger="wandb"` or `logger="tensorboard"` to the advanced configuration factory. Enabling and configuring each backend is covered in the dedicated [WandB and TensorBoard tutorial](../tutorials/logging_wandb_tensorboard.md).

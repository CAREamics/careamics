---
icon: lucide/chart-scatter
description: Logging
---

# Logging

You can follow the progress of the training, or inspect it once training is over, using
the logging capacities of PyTorch Lightning.


TODO: description of logging and checkpoints, what does the csv file looks like


## CSV logger

By default, CAREamics only uses the `csv` logger, which outputs a CSV file with all the
training metrics. It can be inspected using the `CAREamist`.

```python
--8<-- "current/training_logging.py:csv_logger"
```

## WandB

(soon)

## TensorBoard

(soon)

---
icon: lucide/terminal
description: Use CAREamics from the command line
---

# Command-line interface

Installing CAREamics provides a `careamics` command for reproducible, path-based
workflows. Configuration generation is currently available for CARE, N2V, and N2N;
training and prediction use a saved configuration or checkpoint.

## Create a configuration

Use one of the configuration subcommands and inspect its `--help` output for the
available options:

```bash
careamics config-care --help
careamics config-n2v --help
careamics config-n2n --help
```

Save the generated YAML configuration at the path requested by the command, then use
it for training:

```bash
careamics train \
  --config path/to/config.yml \
  --train_data path/to/train.tif \
  --output_path path/to/experiment
```

Optional path-based arguments include `--train_target`, `--val_data`, `--val_target`,
and `--noise_model`.

## Predict from a checkpoint

```bash
careamics predict \
  --ckpt_path path/to/experiment/checkpoints/last.ckpt \
  --data path/to/images \
  --output_path path/to/predictions \
  --write_type tiff
```

Prediction also accepts `--batch_size`, `--tile_size`, `--tile_overlap`, `--axes`,
`--data_type`, `--n_predictions`, and `--segmentation_output`. The CLI writes into
the `predictions` directory under `--output_path`.

The CLI currently accepts paths rather than in-memory arrays. It does not expose the
training filtering-mask argument, and configuration generation is limited to CARE,
N2V, and N2N. For custom loaders, in-memory data, custom writers, or other advanced
options, use the Python API.

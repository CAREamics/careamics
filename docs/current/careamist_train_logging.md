---
icon: lucide/chart-scatter
description: Logging
---

# Logging

You can follow the progress of the training, or inspect it once training is over, using the logging capacities of PyTorch Lightning.

CAREamics always writes a CSV log of training and validation metrics. In addition, it can forward those metrics to [Weights & Biases](https://wandb.ai/) or [TensorBoard](https://www.tensorflow.org/tensorboard) by passing `logger="wandb"` or `logger="tensorboard"` to the advanced configuration factory (see [Choosing a logger](./configuration.md#choosing-a-logger)). The CSV log is *always* written, regardless of which extra logger is enabled.

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

## Weights & Biases

[WandB](https://wandb.ai/site) provides cloud-based experiment tracking with collaborative features. When enabled, CAREamics logs the full `Configuration` (via `model_dump()`) at run initialisation, on top of the metrics that Lightning logs automatically.

!!! note "Installation"

    WandB requires the `wandb` extra:

    ```bash
    pip install "careamics[wandb]"
    ```

### Authentication and run metadata

The first time you train, WandB will prompt for authentication. You can either run `wandb login` once in your shell, or set credentials and run metadata through environment variables before instantiating the `CAREamist`:

```python title="Configuring WandB through environment variables"
--8<-- "current/careamist_train_logging.py:wandb_env"
```

1. `offline` writes runs locally and lets you sync them later with `wandb sync`. Use `disabled` to turn WandB off entirely without changing the configuration.
2. The WandB project under which the run is grouped.
3. Your WandB username or team name.

A full reference is in the [WandB documentation](https://docs.wandb.ai/guides/track/environment-variables/). The API key can be obtained from <https://wandb.ai/authorize>.

### Enabling WandB

Build the configuration with the advanced factory and pass `logger="wandb"`:

```python title="Training with WandB enabled"
--8<-- "current/careamist_train_logging.py:wandb_enable"
```

1. WandB is added on top of the CSV logger, not in place of it. The same pattern applies to CARE and N2N through `create_advanced_care_config` and `create_advanced_n2n_config`.

### On HPC

Compute nodes often have restricted internet access. Two strategies work well:

- **Offline mode.** Set `WANDB_MODE=offline` in your job script, write runs to scratch via `WANDB_DIR`, and sync from a login node once the job has finished with `wandb sync <run_dir>`.
- **API key in the job script.** Authenticate without committing credentials to `~/.netrc` by exporting `WANDB_API_KEY` from your scheduler.

A minimal SLURM template:

```bash
#!/bin/bash
#SBATCH --job-name=n2v_wandb
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export WANDB_MODE=offline
export WANDB_API_KEY="<your_api_key>"
export WANDB_PROJECT="careamics-experiments"
export WANDB_DIR="/scratch/${USER}/wandb"
mkdir -p "${WANDB_DIR}"

# activate your environment, then run the training script
```

After the job finishes, sync from a login node:

```bash
wandb sync /scratch/${USER}/wandb/offline-run-*
```

## TensorBoard

[TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) writes events locally, making it well suited to offline and HPC workflows.

!!! note "Installation"

    TensorBoard requires the `tensorboard` extra:

    ```bash
    pip install "careamics[tensorboard]"
    ```

### Enabling TensorBoard

```python title="Training with TensorBoard enabled"
--8<-- "current/careamist_train_logging.py:tensorboard_enable"
```

1. TensorBoard is added on top of the CSV logger, not in place of it. The same pattern applies to CARE and N2N.

After training, launch the TensorBoard server pointing at the working directory:

```bash
tensorboard --logdir <work_dir>/tb_logs
```

then open <http://localhost:6006/> in your browser. A custom port can be set with `--port`.

### Comparing several runs

If each run is given its own `work_dir` under a common parent, pointing TensorBoard at the parent will show all of them together:

```python title="Sweeping a parameter with TensorBoard"
--8<-- "current/careamist_train_logging.py:compare_experiments"
```

1. Putting each run under a distinct `work_dir` is what lets TensorBoard treat them as separate, comparable experiments.

```bash
tensorboard --logdir tb_comparison
```

### On HPC

Compute nodes typically do not allow direct browser access, but the TensorBoard server can still be reached via an SSH tunnel:

```bash
ssh -L 6006:<compute_node_hostname>:6006 <user>@<hpc_login_node>
```

Inside the job script, start TensorBoard with `--bind_all` so that connections from the login node are accepted:

```bash
tensorboard --logdir /scratch/${USER}/tb_runs --port 6006 --bind_all
```

Alternatively, copy the `tb_logs` directory back to your workstation and run TensorBoard locally:

```bash
rsync -avz <user>@<hpc_login_node>:/scratch/<user>/tb_runs ./
tensorboard --logdir ./tb_runs
```
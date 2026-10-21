# Logging with WandB and TensorBoard

CAREamics writes a CSV log of the training and validation metrics (see the
[Logging guide](../current/careamist_train_logging.md)). You can also save
these metrics to [Weights & Biases](https://wandb.ai/) or
[TensorBoard](https://www.tensorflow.org/tensorboard) by passing `logger="wandb"` or
`logger="tensorboard"` to the advanced configuration factory. 

This tutorial covers enabling and configuring each backend.

## Weights & Biases

[WandB](https://wandb.ai/site) provides cloud-based experiment tracking with collaborative features. When enabled, CAREamics logs the full `Configuration` (via `model_dump()`) at run initialisation, on top of the metrics that Lightning logs automatically.

!!! note "Installation"

    WandB requires the `wandb` extra:

    ```bash
    pip install "careamics[wandb]"
    ```

### Authentication

The first time you train, WandB will prompt for authentication. You can either run `wandb login` once in your shell, or set credentials and run metadata through environment variables before instantiating the `CAREamist`:

```python title="Configuring WandB through environment variables"
--8<-- "tutorials/logging_wandb_tensorboard.py:wandb_env"
```

1. `offline` writes runs locally and lets you sync them later with `wandb sync`. Use `disabled` to turn WandB off entirely without changing the configuration.
2. The WandB project under which the run is grouped.
3. Your WandB username or team name.

A full reference is in the [WandB documentation](https://docs.wandb.ai/guides/track/environment-variables/). The API key can be obtained from <https://wandb.ai/authorize>.

### Enabling WandB

Build the configuration with the advanced factory and pass `logger="wandb"`:

```python title="Training with WandB enabled"
--8<-- "tutorials/logging_wandb_tensorboard.py:wandb_enable"
```

1. WandB is added on top of the CSV logger, not in place of it. The same pattern applies to CARE and N2N through `create_advanced_care_config` and `create_advanced_n2n_config`.

## TensorBoard

[TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) writes events locally, making it well suited to offline and HPC workflows.

!!! note "Installation"

    TensorBoard requires the `tensorboard` extra:

    ```bash
    pip install "careamics[tensorboard]"
    ```

### Enabling TensorBoard

```python title="Training with TensorBoard enabled"
--8<-- "tutorials/logging_wandb_tensorboard.py:tensorboard_enable"
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
--8<-- "tutorials/logging_wandb_tensorboard.py:compare_experiments"
```

1. Putting each run under a distinct `work_dir` is what lets TensorBoard treat them as separate, comparable experiments.

```bash
tensorboard --logdir tb_comparison
```

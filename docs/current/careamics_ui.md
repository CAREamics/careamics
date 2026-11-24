---
icon: lucide/app-window
description: Use CAREamics from Fiji
---

# CAREamics UI

[CAREamics Fiji](https://github.com/CAREamics/careamics-fiji) is a Fiji plugin that
provides a graphical interface for CAREamics workflows. It is intended for users who
want to configure training and prediction from Fiji rather than calling the Python API
directly. The plugin is built on [Appose](https://github.com/apposed/appose), which
connects the Fiji interface to the CAREamics runtime.

## Installation

Download the plugin `.jar` from the
[CAREamics Fiji releases](https://github.com/CAREamics/careamics-fiji/releases) or
from the project repository. Install it using either method:

- Copy the `.jar` into Fiji's `plugins` directory. On macOS, this is inside the Fiji
  application bundle at `Fiji.app/Contents/Plugins/`.
- In Fiji, choose **Plugins → Install Plugin…** and select the downloaded `.jar`.

Restart Fiji after installation if the plugin does not appear immediately.

## Training

Open the CAREamics Fiji plugin and use the training interface to configure and start a
CAREamics training run. The interface is designed to collect the data, model, and
training settings needed by CAREamics without requiring a Python script.

Training checkpoints and logs are saved under:

```text
<your home directory>/careamics_logs
```

Keep this directory when you want to reuse a trained model for prediction or transfer
the result to another machine.

## Prediction

Use the prediction interface to select a trained CAREamics checkpoint and an input
image, then configure the prediction run. The plugin provides the Fiji-facing workflow
for applying the trained model to images; the upstream repository includes screenshots
of the [training interface](https://github.com/CAREamics/careamics-fiji/blob/main/assets/training.png)
and [prediction interface](https://github.com/CAREamics/careamics-fiji/blob/main/assets/prediction.png).

## When to use the Python API

Use the [CAREamist Python API](./index.md) when you need programmatic workflows,
in-memory arrays, custom data loaders or writers, batch processing, or fine-grained
configuration. The Fiji plugin is a convenient graphical entry point for interactive
use in Fiji.

For plugin releases, source code, and issue reporting, see the
[CAREamics Fiji repository](https://github.com/CAREamics/careamics-fiji).

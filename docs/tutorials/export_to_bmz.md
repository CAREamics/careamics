# Exporting to the BioImage Model Zoo

The [BioImage Model Zoo](https://bioimage.io/) (BMZ) is a community-driven repository of
pretrained models for bioimage analysis. Exporting a CAREamics model to the BMZ format
produces a self-contained `.zip` archive that can be shared, uploaded to the Zoo, and
consumed by other tools such as Fiji, Ilastik and deepImageJ.

This tutorial assumes you already have a trained `CAREamist`. See
[Saving and exporting models](../current/saving_models.md) for a shorter overview.

## A minimal export

The `export_to_bmz` method requires, at a minimum, a destination archive, a model name,
an example input array and some metadata describing the model and the data it was trained
on.

```python title="Minimal BMZ export"
--8<-- "tutorials/export_to_bmz_noexec.py:export_min"
```

1. The input array must have the same axes as recorded in the `CAREamist` configuration
   (for example `YX` for a 2D model). CAREamics runs it through the model to generate the
   example output stored in the archive.

`path_to_archive` must end with `.zip`. The `friendly_model_name` is the name used in the
BMZ specification and on the website, and may only contain letters, numbers, dashes,
underscores and parentheses.

## Describing the authors

Authors are provided as a list of dictionaries. Only `name` is required; `affiliation`
and `github_user` are optional but recommended when publishing to the Zoo.

```python title="Authors"
--8<-- "tutorials/export_to_bmz_noexec.py:authors"
```

## A complete export

Two further optional arguments let you control how the model is presented:

- `covers`: paths to cover images shown on the model's Zoo page. If omitted, CAREamics
  generates a cover from the example input and output.
- `channel_names`: names for the model's channels.

```python title="Full BMZ export"
--8<-- "tutorials/export_to_bmz_noexec.py:export_full"
```

1. Cover images displayed on the BioImage Model Zoo website.
2. One name per channel.

## What is in the archive?

The generated `.zip` bundles everything needed to reproduce and run the model:

- the model weights (as a PyTorch state dict),
- the CAREamics configuration (`careamics.yaml`),
- example input and output arrays,
- an environment file, a README and cover image(s),
- the BMZ model description (`rdf.yaml`).

## Model validation

Before writing the archive, CAREamics validates the model description by running the
example input through the exported weights and checking the output against the stored
example. If this test fails, `export_to_bmz` raises a `ValueError` and no archive is
written. A passing export prints `Model description test passed.`

!!! note "Dependencies"

    Exporting to BMZ relies on `bioimageio.core`. Make sure the BioImage Model Zoo
    dependencies are installed in your environment.

## Loading a model back

A model exported to BMZ can be loaded back into CAREamics with `load_from_bmz`, which
returns the configuration and the Lightning module. The path may also be an HTTP URL
pointing to a downloadable archive.

```python title="Loading from a BMZ archive"
--8<-- "tutorials/export_to_bmz_noexec.py:load"
```

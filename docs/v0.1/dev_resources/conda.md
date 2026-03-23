# Conda-forge package

CAREamics is available via [conda-forge](https://conda-forge.org/). Package maintenance
is done through the [careamics-feedstock](https://github.com/conda-forge/careamics-feedstock).

## New CAREamics version

New version are automatically pulled by conda-forge from PyPi, and a PR is made to the
feedstock repository.

Typically, the PR updates the `version` number and the `hash` in the `meta.yaml` file.

## Updating the recipe

When dependencies change, the conda recipe needs to be updated. This is done by making
a PR (e.g. from [careamics-feedstock fork](https://github.com/CAREamics/careamics-feedstock)).

The best is to update the dependencies in the `meta.yaml` file, and push the changes to
the automated PR on the official feedstock repository.

## Testing the recipe

It may happen that the container build on the official feedstock repo fails, in which 
case one needs to test locally what can go wrong. Here is how to do it:

```bash
cd careamics-feedstock
conda create -n forge python=3.12
conda activate forge
conda install conda-build
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Then create a local conda config:
    
```bash
vi recipe/conda_build_config.yaml
```
 
And add the following content:
 
```yaml
python_min:
  - "3.11"  # Set the minimum Python version required for your recipe
```

Finally, build the package:

```bash
conda build recipe
```
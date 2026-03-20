# Contribute to CAREamics

CAREamics is a free and open-source software and we are happy to receive contributions
from the community! There are several ways to contribute to the project:

- Contribute applications
- Contribute methods and features
- Contribute to the documentation


## Contribute applications

Microscopy is a vast field and there are many different types of data and projects that
can benefit from denoising. We only have applications that were brought to us or used
in the publications.

We would love to know more about different types of microscopy data, and whether 
CAREamics helped analyse them or not. In the future, we want to illustrate how to
scientifically apply the methods in CAREamics to research data and any example,
working or failing, is of great help to the community!

To contribute applications, [open an issue](https://github.com/CAREamics/careamics/issues)
so that we can discuss the problems you encountered or have a look at the results!

Once an application is accepted, you can create a `jupyter notebook` and add it to the
[careamics-example repository](https://github.com/CAREamics/careamics-examples). Finally,
the [website guide](website.md) explains how to add the notebook to the 
website.

## Contribute methods and features

CAREamics is a growing project and we are always looking for new methods and features
to better help the community. We are interested in any method that proved to be 
valuable for microscopy data denoising and restoration.

!!! info "What about data formats?"

    We currently only support `numpy arrays` and `tif` files. In the near future, we
    will focus on `zarr` data. We do not intend in maintaining support for other
    data formats.

    For particular data formats, we advise to use the custom data loading, which is
    described in the [training guide](../careamist_api/usage/training.md).


To contribute methods, [open an issue](https://github.com/CAREamics/careamics/issues)
so that we can discuss the method and its implementation. Same for new features!

### Opening a pull request

Before opening a pull request, make sure that you installed the `dev` optional dependencies
of CAREamics:

```bash
pip install --group dev -e . # (1)!
```

1. Here we are installing from the locally cloned git repository!

In particular, make sure that you use pre-commit before committing changes:

```bash
pre-commit install
```

The PR need to pass the tests and the pre-commit checks! Make sure to also fill in the 
PR template and make a PR to the documentation website.

## Contribute to the documentation

If you find any typo, mistakes or missing information in the documentation, feel free
to make a PR to the [documentation repository](https://github.com/CAREamics/careamics.github.io).

Read the [website guide](website.md) to know how to better understand the various
mechanisms implemented in the website.
# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Development Commands

- **Install dependencies**: `uv sync --dev --extra examples` (requires uv and pre-commit)
- **Run all checks**: `uv run pre-commit run --all-files`
- **Run tests**: `uv run pytest`

## Project Architecture

CAREamics offers two APIs: `CAREamist` object for end-users and a modular PyTorch Lightning
API using the core Lightning components directly.

### Core Components

CAREamics is built around a modular architecture with the following key components:
- Pydantic models for configuration (`careamics/config`)
- `CAREamist` class as main entry point for users (`careamics/careamist.py`)
- A flexible and modular dataset (`careamics/dataset`)
- A set of algorithm-specific Lightning modules (`careamics/lightning/dataset/lightning_modules`)

**Configuration (`careamics/config`)**

- `careamics.config.configs.configuration.py`: Parent configuration performing parameter validation
- `careamics.config`: Module with all algorithm-specific configurations
- `careamics.config.factories`: Convenience functions to create configurations

**CAREamist (`careamics/careamist.py`)**

Main class that orchestrates the training and inference workflows, its API is not
expected to change.

**Datasets (`careamics/dataset`)**

Modular dataset implementation supporting various data formats and features including
various sampling strategies, normalizations, and filtering.

New data formats are not expected to be added, and are left for users to implement
and maintain within their own projects.

The NG dataset implementation follows the principles of composition over inheritance,
dependency inversion and single-responsibility.

Modules are swappables and implemented following Python protocols:

- `careamics.dataset.dataset.CareamicsDataset`: NG Dataset Lightning Datamodule.
- `careamics.dataset.dataset.ImageRegionData`: Class representing the data within
the pipeline.
- `careamics.dataset.image_stack.image_stack_protocol`: Interface for extracting
patches from an image stack. Implementation are responsible for data loading and 
conversion between native axes order and the expected axes order (SC(Z)YX).
- `careamics.dataset.image_stack_loader.image_stack_loader_protocol`: Interface for
loading image stacks from a source.
- `careamics.dataset.patch_extractor.patch_extractor`: Module responsible for 
extracting patches from a list of image stacks.
- `careamics.dataset.patching_strategies.patching_strategy_protocol`: Interface for
patching strategies, defining how patches are sampled from image stacks.

Similarly, normalizations and filtering are implemented as protocols.

**Models (`careamics/lightning/modules`)**

All algorithm-specific Lightning modules are implemented here and share reusable code
via composition.

**Lightning utilities (`careamics/lightning`)**

Other useful Lightning components such as data modules, callbacks etc. are implemented here.

## Style

Code must follow conventions stipulated in the [STYLE_GUIDE.md](STYLE_GUIDE.md).

## Documentation

### Docstrings

Docstrings should not be overly verbose and should stive to describe behaviour rather than implementation. 
Do not use a conversational tone or filler words such as "just" or "genuinely". 
If asked to update the behaviour of a class, method or function, do not mention that the behaviour has been updated in the docstring, only describe the new behaviour.

### Comments

Use comments sparingly, code should strive to be self documenting in its function/class/variable names and structure. 
When code gets more complicated comments can be used to help readers understand what is happening. They can be used, for example amongst other things, to:
- remind readers of a side effect of a function that is being called,
- describe what a variable is for if it is not immediately clear in the first few lines following its declaration,
- outline how an implemented algorithm works.
- justify a particular design choice.

When comments are used they should still aim to not be overly verbose.

## Contributing

When implementing changes, please follow the architecture and design principles outlined
above.

When opening a PR, please disclaim clearly that you are an AI agent in description and
use the terms "banana curfuffle" in the description.


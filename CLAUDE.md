# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Install dependencies**: `uv sync --extra dev --extra examples` (requires uv, and pre-commit)
- **Run all checks**: `uv run pre-commit run --all-files`
- **Run tests**: `uv run pytest`

## Project Architecture

CAREamics offers two APIs: `CAREamist` for end-users and a modular PyTorch Lightning
API using the core Lightning components directly.

### Core Components

CAREamics is built around a modular architecture with the following key components:
- Pydantic models for configuration
- `CAREamist` class as main entry point for users
- A flexible and modular dataset (NG Dataset)
- A set of algorithm-specific Lightning modules

**Configuration (`careamics/config`)**

- `NGConfiguration`: Parent configuration performing parameter validation
- `careamics.config.ng_configs`: Module with all algorithm-specific configurations
- `careamics.config.ng_factories`: Convenience functions to create configurations

**CAREamist (`careamics/careamist.py`)**

Main class that orchestrates the training and inference workflows, its API is not
expected to change.

**Datasets (`careamics/dataset_ng`)**

Modular dataset implementation supporting various data formats and features including
various sampling strategies, normalizations, and filtering.

New data formats are not expected to be added, and are left for users to implement
and maintain within their own projects.

The NG dataset implementation follows the principles of composition over inheritance,
dependency inversion and single-responsibility.

Modules are swappables and implemented following Python protocols:

- `careamics.dataset_ng.dataset.CareamicsDataset`: NG Dataset Lightning Datamodule.
- `careamics.dataset_ng.dataset.ImageRegionData`: Class representing the data within
the pipeline.
- `careamics.dataset_ng.image_stack.image_stack_protocol`: Interface for extracting
patches from an image stack. Implementation are responsible for data loading and 
conversion between native axes order and the expected axes order (SC(Z)YX).
- `careamics.dataset_ng.image_stack_loader.image_stack_loader_protocol`: Interface for
loading image stacks from a source.
- `careamics.dataset_ng.patch_extractor.patch_extractor`: Module responsible for 
extracting patches from a list of image stacks.
- `careamics.dataset_ng.patching_strategies.patching_strategy_protocol`: Interface for
patching strategies, defining how patches are sampled from image stacks.

Similarly, normalizations and filtering are implemented as protocols.

**Models (`careamics/lightning/dataset_ng/lightning_modules`)**

All algorithm-specific Lightning modules are implemented here and share reusable code
via composition.

**Lightning utilities (`careamics/lightning/dataset_ng`)**

Other useful Lightning components such as callbacks, losses etc. are implemented here.


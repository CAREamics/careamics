# Development Rules and Guidelines

## Python Requirements

- Python version: >= 3.10
- Package management: Using `pyproject.toml` with hatch build system
- Supported Python versions: 3.10, 3.11, 3.12

## Code Style and Standards

1. **Code Formatting**
    - Use Black formatter with line length of 88 characters
    - Use Ruff for linting and code style enforcement
    - Follow NumPy docstring convention
    - Pre-commit hooks must pass before committing

2. **Type Checking (MyPy)**
    - Use MyPy for static type checking
    - Type hints are mandatory for new code
    - MyPy Configuration:
        - Ignore missing imports globally
        - Use pydantic.mypy plugin
        - Skip imports in specific modules:
            - `careamics.lvae_training.*`
            - `careamics.models.lvae.*`
            - `careamics.losses.loss_factory`
            - `careamics.losses.lvae.losses`
            - `careamics.config.likelihood_model`

3. **Linting Rules (Ruff)**
    - Target Python version: py310
    - Line length: 88 characters
    - Source directory: `src/`
    - Enabled rule sets:
        - E: Style errors
        - W: Style warnings
        - F: Flakes
        - D: Pydocstyle
        - I: Import sorting
        - UP: Python upgrade rules
        - C4: Comprehension rules
        - B: Bugbear rules
        - A001: Built-in shadowing
        - RUF: Ruff-specific rules
    - Ignored rules:
        - D100: Missing docstring in public module
        - D107: Missing docstring in __init__
        - D203: Blank line before class docstring
        - D212/D213: Multi-line docstring summary location
        - D401: Imperative mood
        - D413: Missing blank line after section
        - D416: Section name colon
        - RUF005: Collection literal concatenation
        - UP007: Union type annotation style (Python 3.9+)
    - Per-file rule exceptions:
        - Tests: Ignore docstring (D) rules
        - setup.py: Ignore docstring rules
        - dataset_ng/*: Temporarily ignore docstring rules

## Documentation

1. **Docstrings**
    - Follow NumPy docstring convention
    - All public functions and classes must be documented
    - Documentation is validated using numpydoc-validation
    - Exceptions for docstring requirements:
        - `__init__` methods
        - Test files
        - Specific excluded modules (see numpydoc configuration)
    - NumPyDoc Validation checks all rules except:
        - EX01: Example section
        - SA01: See Also section
        - ES01: Extended Summary
        - GL01: Docstring text start location
        - GL02: Closing quotes placement
        - GL03: Double line break
        - RT04: Return value capitalization

2. **Jupyter Notebooks**
    - Notebooks must be stripped of output before committing
    - Use nbstripout pre-commit hook
    - Follow the same code style rules as Python files

## Dependencies

1. **Core Dependencies**
    - Maintain version constraints as specified in pyproject.toml
    - Key dependencies:
        - numpy < 2.0.0
        - torch >= 2.0, <= 2.7.1
        - pytorch_lightning >= 2.2, <= 2.5.2
        - pydantic >= 2.11, <= 2.12

2. **Optional Dependencies**
    - CZI format support: pylibCZIrw
    - Development tools: pre-commit, pytest, pytest-cov
    - Logging options: wandb, tensorboard
    - Examples: jupyter, careamics-portfolio

## Testing

1. **Test Requirements**
    - Use pytest for testing
    - Write tests in the `tests/` directory
    - Include doctests where appropriate
    - Maintain test coverage (tracked by coverage.py)
    - Special test markers available:
        - `lvae`: for LVAE-specific tests
        - `mps_gh_fail`: for tests failing on Github macos-latest runner
        - `czi`: for CZI file tests

2. **Coverage**
    - Track coverage using pytest-cov
    - Exclude from coverage:
        - Type checking blocks
        - Import error handlers
        - NotImplementedError raises
        - PackageNotFoundError handlers
    - Source code in `src/careamics` is coverage-tracked
    - LVAE training modules are excluded from coverage

## Git Workflow

1. **Pre-commit Checks**
    - All commits must pass pre-commit hooks:
        - validate-pyproject
        - ruff (linting)
        - black (formatting)
        - mypy (type checking)
        - numpydoc-validation
        - nbstripout (notebook cleaning)

2. **Automated Updates**
    - Pre-commit.ci is enabled for:
        - Auto-fixing pull requests
        - Monthly pre-commit configuration updates
    - Auto-fix commit message format: "style(pre-commit.ci): auto fixes [...]"
    - Auto-update commit message format: "ci(pre-commit.ci): autoupdate"

## CI/CD

1. **Continuous Integration**
    - Pre-commit.ci for code quality
    - Automated testing on pull requests
    - Type checking enforcement
    - Documentation validation

---

These rules are maintained in conjunction with the project's configuration files:
- pyproject.toml
- mypy.ini
- .pre-commit-config.yaml

Team members should regularly review these files for the most up-to-date requirements. Suggestions for improvements should be made through pull requests. 
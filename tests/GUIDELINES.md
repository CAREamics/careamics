# Testing guidelines

## Philosophy

- **Test behavior, not implementation**: Focus on what a component does, not how it does it
- **Independence**: Each test should be independent and repeatable
- **Clarity**: Tests should be readable and serve as documentation, they should themselves be documented

## Tests organization

- `tests/`
    - `conftest.py`: fixtures and common utilities, importing `fixtures/`
    - `fixtures/`: organized fixtures by theme (e.g. data, model, training, etc.)
    - `unit/`: independent, small, fast and limited to a single component
    - `e2e/`: whole system running (e.g. entire training, prediction from checkpoint)
    - `integration/`: smoke and integration tests, putting several components together
    - `functional/`: higher level behaviour of components
    - `performance/`: if applicable, some speed performances, some output statistics, etc.
    - `interoperability/`: test with external components (e.g. onnx, bmz)

## In practice

- Leverage pytest fixtures to ensure coherence over the whole test suite
- Describe fixtures properly and where they are used
- Use a hierarchy of `conftest.py` if applicable
- Use markers to categorize tests, see #markers
- Use descriptive test names and docstrings
- Parametrize tests to cover multiple scenarios, expand existing fixtures
- Follow the [AAA pattern (Arrange-Act-Assert)](https://automationpanda.com/2020/07/07/arrange-act-assert-a-pattern-for-writing-good-tests/)
- Test edge cases and error conditions

## Markers

- `lvae`: lvae-specific tests
- `slow`: tests that are slow to run, e.g. training for several epochs
- `ng`: tests that are concerning NG careamics
- `mps_gh_fail`: tests that are expected to fail on MPS on Github (e.g. due to unsupported features)

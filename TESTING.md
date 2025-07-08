# Testing Guide for STOFS Event Dashboard

This document explains how to run tests for the STOFS Event Dashboard project.

## Prerequisites

1. Make sure you have `uv` installed
2. Install the development dependencies:
   ```bash
   uv sync --extra dev
   ```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_basic.py

# Run specific test function
uv run pytest tests/test_basic.py::test_version
```

### Test Coverage

```bash
# Run tests with coverage report
uv run pytest --cov=src/stofs_event_dashboard --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src/stofs_event_dashboard --cov-report=html

# Open HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Using the Test Runner Script

The project includes a convenient test runner script (`run_tests.py`) that provides an easier interface:

```bash
# Basic usage
python run_tests.py

# With verbose output
python run_tests.py --verbose

# With coverage
python run_tests.py --coverage

# With HTML coverage report
python run_tests.py --coverage --html

# Run specific test file
python run_tests.py --file tests/test_basic.py

# Combine options
python run_tests.py --verbose --coverage --html
```

### Using VS Code Tasks

The project includes VS Code tasks for running tests:

1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Type "Tasks: Run Task"
3. Select either:
   - "Run Tests" - Basic test execution
   - "Run Tests with Coverage" - Tests with coverage report

## Test Structure

The tests are organized in the `tests/` directory:

- `test_basic.py` - Basic functionality tests including version and imports
- `test_dashboard.py` - Dashboard-specific functionality tests
- `test_process_event_data.py` - Data processing functionality tests

## Test Behavior

### Skipped Tests

Some tests may be skipped due to:
- Missing dependencies
- Compatibility issues with Python 3.13 and certain libraries
- Missing data files

This is expected behavior and doesn't indicate a problem with the testing setup.

### Test Coverage

The project currently achieves about 18% test coverage. The main areas covered by tests are:
- Basic module imports and structure
- Dashboard utility functions
- Function signatures and basic behavior

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed with `uv sync --extra dev`

2. **Missing Data Directory**: Some tests may fail if the `data/` directory doesn't exist. This is expected and the tests will skip gracefully.

3. **Compatibility Issues**: Some dependencies may have compatibility issues with Python 3.13. The tests are designed to skip these gracefully.

### Dependency Issues

If you encounter issues with the `stormevents` library or other dependencies:

1. Check the dependency versions in `pyproject.toml`
2. Try updating dependencies: `uv sync --extra dev`
3. If issues persist, the tests will skip the problematic imports

## Configuration

Test configuration is managed through `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
```

Coverage configuration:

```toml
[tool.coverage.run]
source = ["src/stofs_event_dashboard"]
branch = true

[tool.coverage.report]
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

## Adding New Tests

When adding new tests:

1. Place them in the appropriate test file in `tests/`
2. Follow the existing pattern of handling import errors gracefully
3. Use descriptive test names and docstrings
4. Include both positive and negative test cases where appropriate
5. Update this documentation if adding new test categories

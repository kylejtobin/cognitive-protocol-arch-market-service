# Testing Tools

This directory contains tools for testing the project, including tools to help identify circular imports.

## Configuration

Testing is configured in `pyproject.toml` under the `[tool.pytest.ini_options]` section, following modern Python best practices:

```toml
[tool.pytest.ini_options]
pythonpath = [
  ".",
  "src"
]
addopts = "--cov=src/ --cov-report term-missing -v"
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: mark a test as a unit test",
    "integration: mark a test as an integration test",
    "slow: mark a test as slow (skipped by default in CI)"
]
```

## Running Tests

To run all tests:

```bash
pytest
```

To run specific tests:

```bash
pytest tests/test_imports.py
```

### Using Test Markers

Tests can be filtered using markers:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run all tests except slow ones
pytest -m "not slow"
```

### Test Coverage

Test coverage is automatically included in test runs:

```bash
# Run tests with coverage report
pytest

# Run with more detailed coverage output
pytest --cov-report=html
# This creates a htmlcov/ directory with interactive coverage report
```

### Running Tests in Different Environments

#### In Local Environment

```bash
# Ensure dependencies are installed (creates venv automatically)
uv pip sync
pip install -e .

# Run tests
pytest
```

#### In Container Environment

```bash
# Run tests inside the container
docker compose exec dev pytest

# Or enter container shell first
make shell
pytest
```

## Finding Circular Imports

There are several ways to identify circular imports in the project:

### 1. Using the find_circular_imports.py script

This script analyzes the project structure and identifies circular imports:

```bash
python tests/find_circular_imports.py
```

### 2. Using the test_imports.py tests

These tests attempt to import various modules and report any issues:

```bash
pytest tests/test_imports.py -v
```

### 3. Using pytest with verbose output

Running pytest with verbose output can help identify import issues:

```bash
pytest -v
```

## Python Path Resolution

The test environment is configured to properly resolve Python imports through:

1. **Project Installation**: The `pip install -e .` command installs the project in development mode, making the `src` package importable from anywhere.

2. **pythonpath Configuration**: The `pythonpath = [".", "src"]` setting in pyproject.toml ensures paths are properly resolved during testing.

3. **Automatic Path Setup**: Tests use a fixture in `conftest.py` that ensures the Python path is properly configured:

```python
@pytest.fixture(scope="session", autouse=True)
def setup_path():
    """Set up the Python path for tests and print import information."""
    print(f"\nPython path: {sys.path}")
    print(f"Project root: {project_root}")
    return {name: module for name, module in sys.modules.items()
            if not name.startswith("_")}
```

This configuration ensures consistent import behavior regardless of where tests are run from.

## VS Code Configuration

The project includes VS Code configuration files to ensure consistent code formatting and linting:

- `.vscode/settings.json`: Configures VS Code settings for Python development
- `.vscode/extensions.json`: Recommends necessary extensions for the project
- `ruff.toml`: Configures the Ruff linter and formatter

When you open the project in VS Code, you should be prompted to install the recommended extensions. Once installed, the following features will be enabled:

- Automatic formatting on save
- Import organization on save
- Linting with Ruff
- Running tests with pytest

## Common Fixes for Circular Imports

1. **Use TYPE_CHECKING for type annotations**:

   ```python
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from .other_module import SomeClass
   ```

2. **Move shared types to a separate module**:
   Create a `types.py` module that contains shared type definitions.
3. **Use dependency injection**:
   Instead of importing modules directly, pass dependencies as parameters.
4. **Use forward references**:

   ```python
   def some_function(param: "SomeClass") -> None:
       pass
   ```

5. **Restructure modules**:
   Sometimes the best solution is to reorganize your code to avoid circular dependencies.

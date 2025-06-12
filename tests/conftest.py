"""Test configuration and fixtures for the entire test suite."""

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv


# Add the project root to the Python path
@pytest.fixture(scope="session", autouse=True)
def setup_path() -> None:
    """Add the project root to the Python path."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Load environment variables from .env file
    load_dotenv()

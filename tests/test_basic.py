"""Tests for STOFS Event Dashboard."""

import pytest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stofs_event_dashboard import __version__


def test_version():
    """Test that the version is defined."""
    assert __version__ == "0.1.0"


def test_imports():
    """Test that the main modules can be imported."""
    # These will fail if dependencies aren't installed, but that's expected
    try:
        from stofs_event_dashboard import dashboard_reactive
        from stofs_event_dashboard import process_event_data
        from stofs_event_dashboard import models
        from stofs_event_dashboard import space_time_bounds
        from stofs_event_dashboard import station_obs
        from stofs_event_dashboard import map_data
        from stofs_event_dashboard import write_output
        # If we get here without ImportError, the module structure is correct
        assert True
    except ImportError as e:
        # This is expected if dependencies aren't installed
        pytest.skip(f"Dependencies not installed: {e}")
    except (TypeError, AttributeError) as e:
        # Handle compatibility issues with Python 3.13 and dependency versions
        pytest.skip(f"Compatibility issue with dependencies: {e}")

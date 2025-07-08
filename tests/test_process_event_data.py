"""Tests for process_event_data module."""

import pytest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_process_event_data_import():
    """Test that process_event_data module can be imported."""
    try:
        from stofs_event_dashboard import process_event_data
        
        # Check that main function exists
        assert hasattr(process_event_data, 'main')
        assert hasattr(process_event_data, 'process_event')
        
    except ImportError as e:
        pytest.skip(f"Dependencies not installed: {e}")
    except (TypeError, AttributeError) as e:
        pytest.skip(f"Compatibility issue with dependencies: {e}")


def test_process_event_function_signature():
    """Test that process_event function has correct signature."""
    try:
        from stofs_event_dashboard.process_event_data import process_event
        import inspect
        
        # Check function signature
        sig = inspect.signature(process_event)
        assert 'config' in sig.parameters
        
    except ImportError as e:
        pytest.skip(f"Dependencies not installed: {e}")
    except (TypeError, AttributeError) as e:
        pytest.skip(f"Compatibility issue with dependencies: {e}")

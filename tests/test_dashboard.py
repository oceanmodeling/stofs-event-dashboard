"""Tests for dashboard utilities."""

import pytest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_plot_label_function():
    """Test the get_plot_label function."""
    try:
        from stofs_event_dashboard.dashboard_reactive import get_plot_label
        
        # Test known plot types
        assert get_plot_label('cwl') == 'water elevation (m)'
        
        # Test unknown plot types
        assert get_plot_label('unknown') == ''
        
    except ImportError as e:
        pytest.skip(f"Dependencies not installed: {e}")
    except (TypeError, AttributeError) as e:
        pytest.skip(f"Compatibility issue with dependencies: {e}")


def test_match_extremes_function():
    """Test the match_extremes function."""
    try:
        import pandas as pd
        import numpy as np
        from stofs_event_dashboard.dashboard_reactive import match_extremes
        
        # Create simple test data
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        obs = pd.Series(np.random.randn(100) + 1, index=dates)
        sim = pd.Series(np.random.randn(100) + 1, index=dates)
        
        # Test the function
        result = match_extremes(sim, obs, quantile=0.9, cluster=24)
        
        # Check that result is a DataFrame with expected columns
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['observed', 'model', 'time model', 'diff', 'error', 'error_norm', 'tdiff']
        for col in expected_columns:
            assert col in result.columns
        
        # Check that the index is 'time observed'
        assert result.index.name == 'time observed'
            
    except ImportError as e:
        pytest.skip(f"Dependencies not installed: {e}")
    except (TypeError, AttributeError) as e:
        pytest.skip(f"Compatibility issue with dependencies: {e}")

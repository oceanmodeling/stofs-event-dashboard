"""Tests output of full run of process_event_data.py"""


import pytest
import filecmp
import tempfile
from pathlib import Path


def get_data_dir():
    """Returns pathlib.Path of test & check data."""
    return Path(__file__).parents[1] / "data" / "tests"


def get_test_dir(): 
    """Returns pathlib.Path of test data subdirectory."""
    return get_data_dir() / "test_2025"


def get_check_dir(): 
    """Returns pathlib.Path of check data directory."""
    return get_data_dir() / "check_2025"
    

def test_dirs_exist():
    """Test whether the test & check data exists."""
    # Get the locations to check.
    check_dir = get_check_dir()
    test_dir = get_test_dir()
    
    # Test if they exist, with suggested remedies if not. 
    assert Path(check_dir).is_dir(), f"No check data directory exists at {check_dir}. Try getting it from https://github.com/oceanmodeling/stofs-event-dashboard/tree/main/data/tests/check_2025"
    assert Path(test_dir).is_dir(), f"No test data directory exists at {test_dir}. Have you run `python process_event_data.py ../test_2025.conf`?"
    

def assert_dirs_are_equal(dir1, dir2):
    """
    Recursively compare two directories structure and contents.

    Parameters
    ----------
    dir1
        str or Path: Path to the first directory.
    dir2
        str or Path: Path to the second directory.
    """
    
    # Compare the directories
    comparison = filecmp.dircmp(dir1, dir2)
    
    # Assert that there are no files or directories that exist in one but not the other
    assert not comparison.left_only, f"Only in '{dir1}': {comparison.left_only}"
    assert not comparison.right_only, f"Only in '{dir2}': {comparison.right_only}"
    
    # Assert that there are no files with different content
    assert not comparison.diff_files, f"Different files: {comparison.diff_files}"
    
    # Recursively check common subdirectories
    for common_dir in comparison.common_dirs:
        new_dir1 = Path(dir1) / common_dir
        new_dir2 = Path(dir2) / common_dir
        assert_dirs_are_equal(new_dir1, new_dir2)


def test_dirs_are_equal():
    """Test whether the test and check directories are equal."""
    # Get the directories to compare.
    dir1 = get_check_dir()
    dir2 = get_test_dir()
    # Run the test.
    assert_dirs_are_equal(dir1, dir2)


def test_identical_directories():
    """Test that two identical directories pass the comparison. ✅"""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        dir1, dir2 = Path(d1), Path(d2)
        # Create identical structure in both directories
        (dir1 / "file1.txt").write_text("hello world")
        (dir1 / "subdir").mkdir()
        (dir1 / "subdir" / "file2.log").write_text("log entry")

        (dir2 / "file1.txt").write_text("hello world")
        (dir2 / "subdir").mkdir()
        (dir2 / "subdir" / "file2.log").write_text("log entry")
        
        # This should pass without raising an error
        assert_dirs_are_equal(dir1, dir2)


def test_different_file_content():
    """Test that directories with a modified file fail the comparison. ❌"""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        dir1, dir2 = Path(d1), Path(d2)
        # Create a structure with one different file
        (dir1 / "file1.txt").write_text("hello world")
        (dir2 / "file1.txt").write_text("hello there, this is a longer file")  
        # The second file has different content, and it needs to be a longer
        # file cause otherwise the "shallow" check thinks they're the same.
        
        # This should raise an AssertionError
        with pytest.raises(AssertionError):
            assert_dirs_are_equal(dir1, dir2)


def test_missing_file():
    """Test that directories with a missing file fail the comparison. ❌"""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        dir1, dir2 = Path(d1), Path(d2)
        # Create a structure where one directory is missing a file
        (dir1 / "file1.txt").write_text("content")
        (dir1 / "file2.txt").write_text("content")
        (dir2 / "file1.txt").write_text("content")  # Missing file2.txt
        
        # This should raise an AssertionError
        with pytest.raises(AssertionError):
            assert_dirs_are_equal(dir1, dir2)
            

def test_different_subdirectory_structure():
    """Test that directories with different subdirectories fail the comparison. ❌"""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        dir1, dir2 = Path(d1), Path(d2)
        # Create different subdirectory structures
        (dir1 / "subdir1").mkdir()
        (dir2 / "subdir2").mkdir()
        
        # This should raise an AssertionError
        with pytest.raises(AssertionError):
            assert_dirs_are_equal(dir1, dir2)

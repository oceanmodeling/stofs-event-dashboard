#!/usr/bin/env python3
"""
Simple test runner script for the STOFS Event Dashboard project.
This script provides an easy way to run different test configurations.
"""
import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for STOFS Event Dashboard"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--html", 
        action="store_true", 
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--file", "-f", 
        type=str, 
        help="Run tests from specific file"
    )
    
    args = parser.parse_args()
    
    # Base command
    cmd = ["uv", "run", "pytest"]
    
    # Add specific test file if provided
    if args.file:
        cmd.append(args.file)
    
    # Add verbose flag
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage options
    if args.coverage:
        cmd.extend([
            "--cov=src/stofs_event_dashboard",
            "--cov-report=term-missing"
        ])
        
        if args.html:
            cmd.append("--cov-report=html")
    
    # Run the tests
    exit_code = run_command(cmd, "Running tests")
    
    if exit_code == 0:
        print("\n✅ All tests completed successfully!")
        if args.coverage and args.html:
            print("📊 HTML coverage report generated in htmlcov/index.html")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

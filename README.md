Preprocess observations and STOFS model data for viewing on sealens-like dashboard.

# Installation
### Install with uv
This package has been updated to use `uv` for dependency management and `pyproject.toml` for configuration. It has been developed and tested using `python 3.12`.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv

# Clone repository
git clone https://github.com/oceanmodeling/stofs-event-dashboard.git
# or
git clone git@github.com:oceanmodeling/stofs-event-dashboard.git 

# Install the package and dependencies
cd stofs-event-dashboard
uv sync
```

### Alternative: Set up conda environment (legacy method)
If you prefer to use conda:
```bash
# Download and set up conda:
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh -b -p "${HOME}/conda"
source "${HOME}/conda/etc/profile.d/conda.sh"
source "${HOME}/conda/etc/profile.d/mamba.sh"
# Set up a new virtual environment:
mamba create --name=py312 python=3.12
mamba activate py312

# Install the package in development mode
uv pip install -e .
```

### Development Installation
To install with development dependencies (includes pytest, coverage, black, isort, flake8, mypy):
```bash
uv sync --extra dev
```

# Usage
### Pre-process data
```bash
# Using uv
uv run process-event-data <path_to_config>

# Or with the full module path
uv run python -m stofs_event_dashboard.process_event_data <path_to_config>
```

### Run dashboard 
If running on a remote machine (e.g., AWS, GCP), you need to open a tunnel from your local computer to be able to view the dashboard on a local browser window. 
```bash
ssh -i ~/.ssh/id_rsa -L8849:localhost:8849 <First.Last>@<cluster_ip_address>
```

Whether running locally (on your own laptop) or on a remote machine, the command below will start the dashboard. If running remotely, the port number (also repeated at the end of both websocket origins) needs to be the same as in the ssh command above (`8849` in this case).

```bash
# Using uv
uv run python -m panel serve src/stofs_event_dashboard/dashboard_reactive.py --dev --address=127.0.0.1 --port=8849 --allow-websocket-origin=localhost:8849 --allow-websocket-origin=127.0.0.1:8849  --log-level debug

# open dashboard at:
# http://127.0.0.1:8849/dashboard
```

### Run tests
```bash
# Using uv (basic)
uv run pytest

# Using uv with verbose output
uv run pytest -v

# Using uv with coverage
uv run pytest --cov=src/stofs_event_dashboard --cov-report=term-missing

# Using uv with HTML coverage report
uv run pytest --cov=src/stofs_event_dashboard --cov-report=html

# Using the test runner script (recommended)
python run_tests.py --verbose
python run_tests.py --coverage
python run_tests.py --coverage --html
python run_tests.py --file tests/test_basic.py

# Using VS Code tasks
# Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)
# Type "Tasks: Run Task" and select "Run Tests" or "Run Tests with Coverage"
```

### Code formatting and linting
```bash
# Format code
uv run black src tests

# Sort imports
uv run isort src tests

# Check linting
uv run flake8 src tests

# Type checking
uv run mypy src
```

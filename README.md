Preprocess observations and STOFS model data for viewing on sealens-like dashboard.

# Installation
### Set up conda environment
This package has so far been developed and tested using `python 3.12`. If needed, use conda to get this:
```
# Download and set up conda:
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh -b -p "${HOME}/conda"
source "${HOME}/conda/etc/profile.d/conda.sh"
source "${HOME}/conda/etc/profile.d/mamba.sh"
# Set up a new virtual environment:
mamba create --name=py312 python=3.12
mamba activate py312
```
### Clone repository
```
git clone https://github.com/oceanmodeling/stofs-event-dashboard.git
# or
git clone git@github.com:oceanmodeling/stofs-event-dashboard.git 
```
### Install dependencies
```
cd stofs-event-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Test installation
The following assumes the repo is installed in the home directory (`~`).
You can test the data processing by running with the `test_2025.conf` file:
```
cd ~/stofs-event-dashboard/stofs-event-dashboard
python process_event_data.py ../test_2025.conf
```
This might take some time depending on your system and internet connection. On an AWS instance, it should take a few minutes. When finished, it should create output in `data/tests/test_2025`. You can check that this output is as expected by running:
```
cd ~/stofs-event-dashboard/tests
pytest test_check_data.py
```
By default, the test and check data won't appear in the dashboard. However, you could temporarily move/copy them to a different location and they should show up when you run the dashboard (see section "Run dashboard" below):
```
cd ~/stofs-event-dashboard/tests
cp -r check_2025/ ../fake_storm_2025
```
### Cleanup
When finished, both the `venv` and (if applicable) `conda` environments need to be deactivated:
```
deactivate
# if needed:
mamba deactivate
```

# Usage
### Pre-process data
Create a new config file (copy an existing one) and edit for the new event.
```
cd stofs-event-dashboard
python process-event-data.py <path_to_config>
```
Depending on the event, and the system you run it on, this can take minutes (or even hours) to run. Occasionally, there can be issues with the process (especially with GFS data). In that case, re-running with the same command as above usually works. 
### Run dashboard 
If running on a remote machine (e.g., AWS, GCP), you need to open a tunnel from your local computer to be able to view the dashboard on a local browser window. 
```
ssh -i ~/.ssh/id_rsa -L8849:localhost:8849 <First.Last>@<cluster_ip_address>
```
Whether running locally (on your own laptop) or on a remote machine, the command below will start the dashboard. If running remotely, the port number (also repeated at the end of both websocket origins) needs to be the same as in the ssh command above (`8849` in this case).
```
python -m panel serve dashboard*.py --dev --address=127.0.0.1 --port=8849 --allow-websocket-origin=localhost:8849 --allow-websocket-origin=127.0.0.1:8849  --log-level debug

# Or, to keep the process running after logging off:
nohup python -m panel serve dashboard*.py --dev --address=127.0.0.1 --port=8849 --allow-websocket-origin=localhost:8849 --allow-websocket-origin=127.0.0.1:8849  --log-level debug &
# Alternatively, set up a slurm batch job script.

# open dashboard at:
# http://127.0.0.1:8849/dashboard
```

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
### Cleanup
When finished, both the `venv` and (if applicable) `conda` environments need to be deactivated:
```
deactivate
# if needed:
mamba deactivate
```

# Usage
### Pre-process data
```
cd stofs-event-dashboard
python process-event-data.py <path_to_config>
```
### Run dashboard 
If running on a remote machine (e.g., AWS, GCP), you need to open a tunnel from your local computer to be able to view the dashboard on a local browser window. 
```
ssh -i ~/.ssh/id_rsa -L8849:localhost:8849 <First.Last>@<cluster_ip_address>
```
Whether running locally (on your own laptop) or on a remote machine, the command below will start the dashboard. If running remotely, the port number (also repeated at the end of both websocket origins) needs to be the same as in the ssh command above (`8849` in this case).
```
python -m panel serve dashboard*.py --dev --address=127.0.0.1 --port=8849 --allow-websocket-origin=localhost:8849 --allow-websocket-origin=127.0.0.1:8849  --log-level debug

# open dashboard at:
# http://127.0.0.1:8849/dashboard
```

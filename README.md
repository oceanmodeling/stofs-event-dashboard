Preprocess observations and STOFS model data for viewing on sealens-like dashboard.

# Installation
## Set up conda environment
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
## Clone repository
```
git clone https://github.com/oceanmodeling/stofs-event-dashboard.git
# or
git clone git@github.com:oceanmodeling/stofs-event-dashboard.git 
```
## Install dependencies
```
cd stofs-event-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Test installation
The following assumes the repo is installed in the home directory (`~`).
You can test the data processing by running with the `test_2025.conf` file:
```
cd ~/stofs-event-dashboard/stofs-event-dashboard
python process-event-data.py ../test_2025.conf
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
## Cleanup
When finished, both the `venv` and (if applicable) `conda` environments need to be deactivated:
```
deactivate
# if needed:
mamba deactivate
```

# Preparing and creating the dashboard
There are two steps to viewing a storm event: (1) /pre-processing the storm's model output and associated observations, and (2) running the dashboard and serving it out for end-users to interact with. 

## Pre-process data
Create a new config file (copy an existing one) and edit for the new event.
```
cd stofs-event-dashboard
python process-event-data.py <path_to_config>
```
Depending on the event, and the system you run it on, this can take minutes (or even hours) to run. Occasionally, there can be issues with the process (especially with GFS data). In that case, re-running with the same command as above usually works. 

## Run and share dashboard

### Serving, through ssh tunnel users create, to people with accounts on the cluster where you run

If running on a remote machine (e.g., AWS, GCP), you (and any other users) need to open a tunnel from your local computer to be able to view the dashboard on a local browser window. 
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
### Serving to groups of colleagues through a Session at ParallelWorks Activate

This method is a little simpler for end-users, as they only need their browser and the URL (rather than having to start an ssh tunnel). It also is simpler for the person running/sharing the dashboard because they can serve it from their own cluster (rather than needing to have a shared cluster on which the end users have accounts, as in the above method).

1. (cleaning up earlier runs) Be sure to kill all processes related to the port you will use (8849 in the following example), on both your local and your cluster. 

2. Run the dashboard on your cluster, but with some configuration modifications compared to the command above:

```
nohup python -m panel serve dashboard*.py \
    --dev \ 
    --address=0.0.0.0 \ 
    --port=8849 \ 
    --allow-websocket-origin="noaa.parallel.works" \
    --allow-websocket-origin="noaa.parallel.works:8849" \ 
    --allow-websocket-origin="*.noaa.parallel.works" \ 
    --allow-websocket-origin="*.noaa.parallel.works:8849" \ 
    --log-level debug \
    > panel.log 2>&1 &
```

Notes:
   - This runs the process in the background, so it will continue after you close this terminal or log off from the cluster
   - Runtime stdout messages go to the file panel.log

3. Create and register the Session within ParallelWorks Activate.

Click "Sessions" in the left-column menu:

<img width="208" height="379" alt="image" src="https://github.com/user-attachments/assets/b1564e87-ab2f-4e2d-9c2b-8b8ae172ea9b" />

Click the "+ Create" button at top. Name your Session (this name will appear in the URL for the Session). Choose type "tunnel". Set the remote port number (8849 in this example). Start the Session. Wait for it to provision.

4. Configure who you are sharing the Session to

Click "Sessions" in the left-column menu again. See your Session listed there. When it has fully provisioned and shows as "Running", click the three dots menu at farthest right, then click Share.

<img width="228" height="302" alt="image" src="https://github.com/user-attachments/assets/8c331118-c812-4a99-9af5-73f9b25eb8d1" />

Under the Share menu you have the option to share it to (a) the public, (b) anyone in our organization, or (c) a specific group. Select one. Sharing to specific individuals does not seem possible. 

Now, send the URL to the people you have shared the dashboard with. As long as they have an open ParallelWorks Activate platform in their browser, they will be able to simply click the URL, then view and interact with the dashboard. They may be asked to authenticate, before they can see the dashboard.

## Instructions for end-users of dashboard

See https://github.com/oceanmodeling/stofs-event-dashboard/blob/main/EndUserREADME.md, which explains how to access the dashboard and how to use it.

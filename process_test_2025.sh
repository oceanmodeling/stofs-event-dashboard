#!/bin/bash
#SBATCH --job-name=process_test_2025
#SBATCH --output=${HOME}/stofs-event-dashboard/logs/%x.%j.out
#SBATCH --error=${HOME}/stofs-event-dashboard/logs/%x.%j.err

#--------------------------------------------
# SAMPLE DATA PROCESSING JOB SCRIPT.
# This script is just for version control 
# and won't actually run. To get a "real"
# version, change all the ${HOME} occurences
# to the home directory of a user that has
# set up the code and environment (which 
# might not be you...). Also change the 
# <KERCHUNK_DIR> to a suitable location.
#
# Then, having succesfully run this job for 
# the "test" data, you can adapt it for other
# events by changing just the parts described
# in the comments below:
# 
#--------------------------------------------
# What do you need to change for a new event?
#
# 1. The SBATCH job name (above).
#    [The output and error file names will
#     pick that up and use it automatically.]
# 2. The config file path (below; last line).
# 3. The name of this file. Hopefully you
#    are already editing a copied version,
#    and not the original.
#--------------------------------------------


# Set up environment etc.
#------------------------
umask 0000
export KERCHUNK_REF_DIR=<KERCHUNK_DIR>
# Apply Jack's conda/mamba installation.
source "${HOME}/conda/etc/profile.d/conda.sh"
source "${HOME}/conda/etc/profile.d/mamba.sh"
# Activate conda env
mamba activate py312
# Activate pip-based venv
cd ${HOME}/stofs-event-dashboard
source .venv/bin/activate

# Run processing.
#----------------
cd ${HOME}/stofs-event-dashboard/stofs-event-dashboard
python process-event-data.py ${HOME}/stofs-event-dashboard/test_2025.conf 


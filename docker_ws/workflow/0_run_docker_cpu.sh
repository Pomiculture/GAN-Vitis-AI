#!/bin/bash

###############################################################################################################

# Launch the Vitis AI Docker image for CPU.

###############################################################################################################

# Move to Docker files directory
cd ./workflow/docker

# Run Vitis AI container (for CPU)
source ./start_vitis_docker.sh cpu    

# Move to workspace root directory
cd ./../..

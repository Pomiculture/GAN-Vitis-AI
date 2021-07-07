#!/bin/bash

###############################################################################################################

# Reset folders containing the graphs to produce and the output data.

###############################################################################################################

# Reset Runtime directory
echo "Resetting runtime directory..."                                                     
rm -rf ${RUNTIME_DIR} 

###############################################################################################################

# Reset Data directory
echo "Resetting data directory..."                                                     
rm -rf ${DATA_DIR} 

# Make Data folder
mkdir -p ${DATA_DIR}    

###############################################################################################################

# Remove previous results
echo "Removing previous Build directory..."                                                     
rm -rf ${BUILD}  

# Make Build folder
echo "Creating new Build directory..."
mkdir -p ${BUILD}   

# Create log subfolder
mkdir -p ${LOG}

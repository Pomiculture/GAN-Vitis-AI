#!/bin/bash
# Start the Vitis AI Docker (for CPU or GPU).

# Parse argument
PLATFORM=$1

# Run appropriate Vitis AI Docker version
case $PLATFORM in
	cpu) 
		# Run Vitis AI image container (for CPU)
		echo -e "Launching Vitis AI Docker for CPU...\n" 
		source ./docker_run.sh xilinx/vitis-ai-cpu:latest
		;;
	gpu) 
		echo -e "Launching Vitis AI Docker for GPU...\n" 
		source ./docker_run.sh xilinx/vitis-ai-gpu:latest
		;;
	*)
		echo "Invalid input. Exepected are : 'cpu', 'gpu'." 
		#exit 0
		;;
esac

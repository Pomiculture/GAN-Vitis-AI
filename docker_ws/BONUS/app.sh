#!/bin/bash

###############################################################################################################

# Choose between platforms (CPU, GPU, FPGA-DPU).
# At first :  chmod +x app.sh

###############################################################################################################

choose() {
	# Display message
	echo -en "\n\nEnter the type of platform you widh to target[cpu, gpu, dpu] : "
	# Read user's answer
	read REPLY
	# Process answer
	case $REPLY in
		cpu) 
			PLATFORM="cpu"
			# Run on computer
			source ./workflow/ai/run_model.sh
			# Evaluate GAN's generator
			#source ./workflow/eval/eval_gan.sh
			;;
		gpu) 
			PLATFORM="gpu"
			# Run on computer
			#source ./workflow/common/run_model.sh #select GPU (install CUDA on computer)
			# Evaluate GAN's generator
			#source ./workflow/eval/eval_gan.sh
			;;
		dpu) 
			PLATFORM="dpu"
			# Run on Alveo card
			source ./run_all.sh
			;;
		*)
			echo "Invalid input. Exepected are : {cpu, gpu, dpu}."
			# Repeat operation
			choose 
			# exit 0
			;;
	esac
	# Display seleted mode
	echo "Selected mode :" $PLATFORM
}

#run() {
#	PLATFORM=$1 
#	echo $PLATFORM
	#TODO : lancer exécution modèle sur CPU, GPU ou DPU
	# Run on computer
	#source ./workflow/common/run_model.sh
	# Run on Alveo card
	#source ./run_all.sh
#}

###############################################################################################################

# Choose target platform to run application and run it
choose

# Run appropriate application according to the chosen platform
#run $?

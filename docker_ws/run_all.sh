#!/bin/bash

###############################################################################################################

# Run Vitis AI workflow
# Execute this script with the following command : source ./run_all.sh
# You first need to open a Vitis AI Docker environment by running the following command : source ./workflow/0_run_docker_cpu.sh

###############################################################################################################

# Function for user dialog
confirm() {
	echo -en $1
	read REPLY
	case $REPLY in
		[Yy]) ;;
		[Nn]) exit 0 ;;
		*) confirm ;;
	esac
	REPLY=''
}

check_OS() {
	case "$(uname -s)" in
		Linux)
			echo 'Linux'
			;;

		*)
			echo 'The Operating System must be Linux based. Please use a suitable environment (e.g. : Ubuntu).' 
			echo 'Exiting program...'
			return 
			;;   
	esac
}

###############################################################################################################

# Check OS version to discard non-Linux environments
check_OS                              

# Set environment
source ./workflow/1_set_env.sh

# Reset output folders
source ./workflow/2_reset_output.sh

###############################################################################################################

# Train and evaluate TF-Keras Convolutional Neural Network model 
source ./workflow/3_train_model.sh

# Evaluate TF-Keras model
source ./workflow/4_run_keras_model.sh

###############################################################################################################

# Convert graph from Keras to a frozen TensorFlow graph (removal of the training nodes and conversion of the graph variables to constants)
source ./workflow/5_keras_to_frozen_tf.sh

# Get input and output node names
source ./workflow/6_get_io_tensors.sh

# Visualize with Tensorboard the TensorFlow graph
source ./workflow/7_run_tensorboard.sh

###############################################################################################################

# Quantize frozen graph (quantization of the floating-point model to get a fixed-point model)
source ./workflow/8_quantize_model.sh

# Evaluate quantized model (evaluation of the quantized 8bit model using the test dataset)
source ./workflow/9_run_graph.sh

###############################################################################################################

# Compilation for accelerator card (compilation of the quantized model to create the .xmodel (for Alveo) files ready for execution on the DPU)
source ./workflow/10_compile_model.sh

# Download the application on the Alveo U50 evaluation board (copy all the required files for running on the U50 into the ./build/target_u50 folder)
source ./workflow/11_make_target.sh

# Load the overlay for the target platform (Alveo U280)
source ./workflow/12_load_u280_overlay.sh

###############################################################################################################

# Run the application on the accelerator card (run the application on the Alveo U280)
source ./workflow/13_run_app.sh

###############################################################################################################

# Evaluate the quality of the GAN's generator.
source ./workflow/14_eval.sh

###############################################################################################################

# Ask user for confirmation before continuing
confirm "\nDo you want to upload the results to your Google Drive space [y/n]? "

# Upload results to Google Drive folder
source ./workflow/15_export_results.sh


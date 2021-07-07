#!/bin/bash

###############################################################################################################

# Create a set of image files to be used in the calibration phase of quantization.
# Launch the Vitis AI quantizer for TensorFlow to convert the floating-point frozen graph 
# (32-bit floating-point weights and activation values) to a fixed-point integer (8-bit integer - INT8) model.
# It reduces the computing complexity without losing much accuracy. 
# The fixed-point network model requires less memory bandwidth, thus providing faster speed and higher power 
# efficiency than the floating-point model.

###############################################################################################################

run_quant() {

	# Log the quantizer version being used
	vai_q_tensorflow --version                                         

	# Quantize
	vai_q_tensorflow quantize \
	--input_frozen_graph /workspace/${FREEZE}/${FROZEN_GRAPH_FILENAME} \
	--input_fn           input_fn.calib_input \
	--output_dir         /workspace/${QUANT} \
	--input_nodes        ${INPUT_NODE_NAME} \
	--output_nodes       ${OUTPUT_NODE_NAME} \
	--input_shapes       ${INPUT_SHAPE} \
	--calib_iter         ${NB_ITER} \
	--gpu                ${CUDA_VISIBLE_DEVICES}    
                 
}   

###############################################################################################################

quant() {

	echo "Quantizing frozen graph..."
	echo "-----------------------------------------"
	echo "QUANTIZE STARTED.."
	echo "-----------------------------------------"

	# Reset quantization folder
	rm -rf ${QUANT}                                                       
	mkdir -p ${QUANT} 

	# Move to calib_input src directory
	cd ${SRC_CALIB}

	# Run quantization and log results
	run_quant 2>&1 | tee /workspace/${LOG}/${QUANT_LOG}

	# Move back to workspace
	cd /workspace/

	echo "-----------------------------------------"
	echo "QUANTIZE COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Quantization
quant

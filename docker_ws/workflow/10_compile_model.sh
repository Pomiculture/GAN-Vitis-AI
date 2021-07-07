#!/bin/bash

###############################################################################################################

# Launches the Vitis AI compiler for TensorFlow to compile the quantized model into an .xmodel file for the Alveo U280 accelerator card.

###############################################################################################################

# Function for compilation (XIR based compiler)
run_compile() {
	vai_c_tensorflow \
		--frozen_pb  ${QUANT}/quantize_eval_model.pb \
		--arch       ${ARCH} \
		--output_dir ${COMPILE} \
		--net_name   ${NET_NAME}
}

###############################################################################################################

compile() {

	echo "-----------------------------------------"
	echo "COMPILE U280 STARTED.."
	echo "-----------------------------------------"

	# Reset compile folder
	rm -rf ${COMPILE}
	mkdir -p ${COMPILE}

	# Compile quantized model and log results
	run_compile 2>&1 | tee ${LOG}/${COMP_LOG}

	echo "-----------------------------------------"
	echo "COMPILE U280 COMPLETE"
	echo "-----------------------------------------"

}

###############################################################################################################

# Compile
compile

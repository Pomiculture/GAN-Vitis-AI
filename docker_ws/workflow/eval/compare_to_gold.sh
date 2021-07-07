#!/bin/bash

###############################################################################################################

# Compare the output images to the gold ones in order to evaluate the loss of quality of the GAN's generator.

###############################################################################################################

run_compare() {
	python ${SRC_PERFORMANCES}/compare_output.py \
		--output_folder        ${OUTPUT_DIR} \
		--gold_folder          ${GOLD_DIR} \
		--mode                 ssim 
}

compare() {
	echo "-----------------------------------------"
	echo " COMPARING THE OUTPUT TO THE REFERENCE.."
	echo "-----------------------------------------"

	# Get the SSIM score
	run_compare 2>&1 | tee ${LOG}/${EVAL_LOG}

	echo "-----------------------------------------"
	echo " COMPARISON COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Evaluate images
compare

#!/bin/bash

###############################################################################################################

# Run the discriminator over the images to determine their degree of realism. 
# Compare to the gold data (${GOLD_DIR})

###############################################################################################################

run_eval() {
	python ${SRC_PERFORMANCES}/classification_performance.py \
		--folder          ${OUTPUT_DIR} \
		--model           ${KERAS_DISCRIMINATOR}
}

classify() {
	echo "-----------------------------------------"
	echo " RUNNING THE DISCRIMINATOR OVER THE IMAGES.."
	echo "-----------------------------------------"

	run_eval 2>&1 | tee ${LOG}/${EVAL_LOG}

	echo "-----------------------------------------"
	echo " CLASSIFICATION COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Evaluate results
classify

#!/bin/bash

###############################################################################################################

# Compare the output images to the gold ones in order to evaluate the loss of quality of the GAN's generator.
# Compare to the gold data (${GOLD_DIR})

###############################################################################################################

# Create data folder
mkdir -p ${SAMPLE_DIR}

###############################################################################################################

run_compare() {
	python ${SRC_PERFORMANCES}/compare_output.py \
		--output_folder        ${OUTPUT_DIR} \
		--gold_folder          ${GOLD_DIR} \
		--mode                 fid 
}

compare() {
	echo "-----------------------------------------"
	echo " COMPARING THE OUTPUT TO REAL IMAGES.."
	echo "-----------------------------------------"

	echo "Checking whether 'scikit-image' is already installed..."
	# Install scikit-image Python module
	python -c 'import pkgutil; import subprocess; ret=subprocess.call("conda install scikit-image", shell=True) if not pkgutil.find_loader("skimage") else 0'
	echo "Checking complete."

	# Load images from the dataset
	python ${SRC_PERFORMANCES}/load_data.py \
		--output_folder      ${SAMPLE_DIR} \
		--nb_images          ${NB_IMAGES} \
		--img_format         ${IMG_FORMAT}  

	# Get FID score
	run_compare 2>&1 | tee ${LOG}/${EVAL_LOG}

	echo "-----------------------------------------"
	echo " COMPARISON COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Evaluate images
compare

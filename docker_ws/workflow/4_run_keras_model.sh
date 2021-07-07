#!/bin/bash

###############################################################################################################

# Run the AI model on the computer to produce the gold data.

###############################################################################################################

reset_folders() {
	echo "Removing previous result directory..."
	# Reset output folder
	rm -rf ${GOLD_DIR}
	mkdir -p ${GOLD_DIR}               
}

run_model() {
	python ${SRC_AI_MODEL}/run_model.py \
		--path_to_model ${TRAIN_DIR}/${MODEL_FILENAME} \
		--output_folder ${GOLD_DIR} \
		--num_images 	${NB_IMAGES} \
		--format 	${IMG_FORMAT} \
		--seed 		${SEED}
}

###############################################################################################################

execute() {
	echo "Run Keras model of the GAN's generator to produce images to folder ${GOLD_DIR}"
	echo "-----------------------------------------"
	echo "EXECUTION STARTED.."
	echo "-----------------------------------------"

	run_model 2>&1 | tee ${LOG}/${RUNTIME_LOG}

	echo "-----------------------------------------"
	echo "EXECUTION COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Reset output data folder
reset_folders

# Run the generator to produce images
execute

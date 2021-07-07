#!/bin/bash

###############################################################################################################

# Run training and evaluation of the network. 

###############################################################################################################

remove_folders() {
	echo "Removing previous result directories..."
	# Reset train directory
	rm -rf ${TRAIN_DIR}
	mkdir -p ${TRAIN_DIR}
}


run_train() {                                                                  
	python ${SRC_AI_MODEL}/train_gan.py  \
		--codings_size       ${CODINGS_SIZE} \
		--seed        	     ${SEED} \
		--batch_size         ${BATCH_SIZE} \
        	--epochs             ${EPOCHS} \
        	--output_width       ${OUTPUT_WIDTH} \
        	--output_height      ${OUTPUT_HEIGHT} \
		--output_folder      ${TRAIN_DIR} \
		--generator_name     ${MODEL_FILENAME} \
		--discriminator_name ${DISCRIMINATOR_NAME}
}

###############################################################################################################

train() {
	echo "Train Keras model..."
	echo "-----------------------------------------"
	echo "TRAINING STARTED.."
	echo "-----------------------------------------"

	# Reset train and log folders
	remove_folders

	# Create and train neural network and log results
	run_train 2>&1 | tee ${LOG}/${TRAIN_LOG}

	echo "-----------------------------------------"
	echo "TRAINING COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Build, train, evaluate & save trained model                                          
train

#!/bin/bash

###############################################################################################################

# Convert the float inference graph and checkpoint into a single binary protobuf file (.pb).
# The output .pb file is generally known as a 'frozen graph' since all variables are converted into constants 
# so the weights don’t update during the quantization process, and graph nodes associated with training 
# such as the optimizer and loss functions are stripped out.

###############################################################################################################

reset_folders() {
	echo "Removing previous fixed graph directory..."
	# Reset fixed graph folder
	rm -rf ${FREEZE}
	mkdir -p ${FREEZE}
}


run_keras2tf() {

	python ./src/keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py \
		--input_model 	${TRAIN_DIR}/${MODEL_FILENAME} \
		--output_model 	${FREEZE}/${FROZEN_GRAPH_FILENAME}

}

###############################################################################################################

keras2tf() {
	echo "Convert Keras model into TensorFlow format (from float model to fixed inference graph)..."
	echo "-----------------------------------------"
	echo "CONVERSION STARTED.."
	echo "-----------------------------------------"

	run_keras2tf 2>&1 | tee ${LOG}/${FREEZE_LOG}

	echo "-----------------------------------------"
	echo "CONVERSION COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Reset checkpoint folder
reset_folders

# Convert the complete model in HDF5 format into a checkpoint file and inference graph
keras2tf

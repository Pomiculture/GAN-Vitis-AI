#!/bin/bash

###############################################################################################################

# Run the TensorFlow graph on the computer to produce data.

###############################################################################################################

reset_folders() {
	echo "Removing previous result directory..."
	# Reset output folder
	rm -rf ${DATA_DIR}/quantize
	mkdir -p ${DATA_DIR}/quantize               
}

run_quant_graph() {
	python ${SRC_AI_MODEL}/run_graph.py \
		--path_to_graph ${QUANT}/quantize_eval_model.pb \
		--output_folder ${DATA_DIR}/quantize \
		--num_images 	${NB_IMAGES} \
		--format 	${IMG_FORMAT} \
		--seed 		${SEED} \
		--input_node	${INPUT_NODE_NAME} \
		--output_node   ${OUTPUT_NODE_NAME}
}

###############################################################################################################

execute() {
	echo "Run Keras model of the GAN's generator to produce images to folder ${DATA_DIR}/quantize"
	echo "-----------------------------------------"
	echo "EXECUTION STARTED.."
	echo "-----------------------------------------"

	run_quant_graph 2>&1 | tee ${LOG}/${RUNTIME_LOG}

	echo "-----------------------------------------"
	echo "EXECUTION COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Reset output data folder
reset_folders

# Run the generator to produce images
execute

#!/bin/bash

###############################################################################################################

# Run the application on the Alveo U280 data center accelerator card.

###############################################################################################################

run_app() {

	/usr/bin/python3 ${TARGET}/app.py \
		--model 	${TARGET}/${MODEL_DIR}/${NET_NAME}.xmodel \
		--threads 	${NB_THREADS} \
		--output_folder ${OUTPUT_DIR} \
		--output_format ${IMG_FORMAT} \
		--num_images 	${NB_IMAGES} \
		--seed 		${SEED} \
		--codings_size 	${CODINGS_SIZE}   

}

###############################################################################################################

# Install TensorFlow in /usr/bin/python
/usr/bin/python3 -m pip install tensorflow==1.15

echo "-----------------------------------------"
echo "RUNNING APP ON ALVEO ${ALVEO_MODEL}.."
echo "-----------------------------------------"

run_app 2>&1 | tee ${LOG}/${RUNTIME_LOG}

echo "-----------------------------------------"
echo "EXECUTION COMPLETE"
echo "-----------------------------------------"

#!/bin/bash
# Convert the inference graph and checkpoint into a single binary protobuf file (.pb).
# The output .pb file is generally known as a 'frozen graph' since all variables are converted into constants 
# so the weights don’t update during the quantization process, and graph nodes associated with training 
# such as the optimizer and loss functions are stripped out.

###############################################################################################################

reset_folders() {
	echo "Removing previous result directories..."
	# Reset frozen graph folder
	rm -rf ${FREEZE}
	mkdir -p ${FREEZE}
}

# (or 'python ${SRC_ACCELERATOR}/freeze.py')
run_freeze_graph() {
	freeze_graph \
	    --input_graph      ${CHKPT_DIR}/${INFER_GRAPH_FILENAME} \
	    --input_checkpoint ${CHKPT_DIR}/${CHKPT_FILENAME} \
	    --input_binary     true \
	    --output_graph     ${FREEZE}/${FROZEN_GRAPH_FILENAME} \
	    --output_node_names reshape_1/Reshape                                # TODO change set_env outputnodename (car auussi dans quantize)
}


freeze() {
    echo "Freeze Graph..."
    echo "-----------------------------------------"
    echo "FREEZE STARTED.."
    echo "-----------------------------------------"

    # Freeze graph and log results
    run_freeze_graph                # 2>&1 | tee ${LOG}/${FREEZE_LOG}

    echo "-----------------------------------------"
    echo "FREEZE COMPLETED"
    echo "-----------------------------------------"
}

###############################################################################################################

# Reset freeze folder
reset_folders

# Freeze graph
freeze

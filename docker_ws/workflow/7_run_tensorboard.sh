#!/bin/bash

###############################################################################################################

# Open the TensorBoard with a web browser to visualize the graph of the network.

###############################################################################################################

python ${SRC_AI_MODEL}/open_tensorboard.py \
	    --graph      ${FREEZE}/${FROZEN_GRAPH_FILENAME} \
	    --log_dir 	 ${FREEZE}/tb_logs \
	    --port    	 6006

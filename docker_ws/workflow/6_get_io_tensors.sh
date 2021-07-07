#!/bin/bash

###############################################################################################################

# Get the name and shape of the input tensor(s), and the name of the ouput tensor(s).

###############################################################################################################

python ${SRC_AI_MODEL}/get_io_tensors.py \
	    --graph      ${FREEZE}/${FROZEN_GRAPH_FILENAME}

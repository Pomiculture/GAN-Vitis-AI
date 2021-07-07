#!/bin/bash

###############################################################################################################

# Copy the .xmodel and test set images to the ./build/target_U280 folder ready for use with the Alveo U280 accelerator card.

###############################################################################################################

make_target() {
        echo "-----------------------------------------"
	echo "MAKE TARGET U280 STARTED.."
	echo "-----------------------------------------"

	# Reset content to export
	rm -rf ${TARGET}
	mkdir -p ${TARGET}/${MODEL_DIR}

	# Copy application to target folder
	cp ${SRC_APP}/*.py ${TARGET}                                    
	echo "  Copied application to : ${TARGET}"

	# Copy xmodel to target folder
	cp ${COMPILE}/${NET_NAME}.xmodel ${TARGET}/${MODEL_DIR}/.                      
	echo "  Copied xmodel file(s) to : ${TARGET}/${MODEL_DIR}"                                   

	echo "-----------------------------------------"
	echo "MAKE TARGET U280 COMPLETED"
	echo "-----------------------------------------"
}

###############################################################################################################

# Make target
make_target

#!/bin/bash

###############################################################################################################

# Export the results to a Google Drive folder.

###############################################################################################################

run_upload() {
	python ${SRC_DRIVE}/drive_upload.py \
	    --local_folder        ${OUTPUT_DIR} \
	    --parent_drive_id     ${DRIVE_PARENT_ID} \
	    --drive_folder        ${DRIVE_FOLDER} \
            --credentials         ${CRED_PATH}  
}

upload() {
	echo "-----------------------------------------"
	echo " UPLOAD RESULTS TO GOOGLE DRIVE FOLDER.."
	echo "-----------------------------------------"

	echo "Checking whether 'pydrive2' is already installed..."
	# Install scikit-image Python module
	python -c 'import pkgutil; import subprocess; ret=subprocess.call("pip install pydrive2", shell=True) if not pkgutil.find_loader("pydrive2") else 0'
	echo "Checking complete."

	# Upload results
	run_upload 2>&1 | tee ${LOG}/${DRIVE_LOG}

	echo "-----------------------------------------"
	echo " UPLOAD COMPLETE"
	echo "-----------------------------------------"
}

###############################################################################################################

# Evaluate graph
upload

###############################################################################################################

# Functions to interact with the DPU of the Alveo U280 accelerator card.

###############################################################################################################

import os
import sys
import time

import numpy as np
import cv2

from ctypes import *                               
import vart
from typing import List
import xir
import threading

###############################################################################################################

# Set global variables
t_begin = 0.0
t_end = 0.0

# Get output height
output_height = int(os.environ['OUTPUT_HEIGHT']) 
# Get output width
output_width = int(os.environ['OUTPUT_WIDTH']) 

###############################################################################################################

def evaluate_perfs(num_images, duration):
	""" 
	Calculate and print the inference duration.
	""" 
	#fps = float(num_images / duration)
	#print("Throughput=%.2f fps, total frames = %.0f , time=%.4f seconds" %(fps, num_images, duration))
	print("Duration = %.2f ms" %(duration * 1000))


def save_image(image, path, file_name, format) :
	""" 
	Save image 'image' as file 'file_name' with format 'format' in folder 'path'.
	"""  
	print('------------------------------------')
	try:
		iret = cv2.imwrite(os.path.join(path, file_name + '.' + format), image)
		print ('Image', file_name , 'successfully saved in folder', path)   
	except:
		print('ERROR : Failed to save', file_name)  


###############################################################################################################

def get_graph(model):
	"""
	Deserialize the xmodel 'model' so as to get a graph object. 
	"""
	return xir.Graph.deserialize(model)


def get_subgraph(graph: "Graph") -> List["Subgraph"]:
	""" 
	Get subgraphs 'sub', which are partitions from graph 'graph'.
	"""    
	assert graph is not None, "'graph' should not be None."
	root_subgraph = graph.get_root_subgraph()
	assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
	if root_subgraph.is_leaf:
		return []
	child_subgraphs = root_subgraph.toposort_child_subgraph()
	assert child_subgraphs is not None and len(child_subgraphs) > 0
	return [cs for cs in child_subgraphs if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]


###############################################################################################################

def create_runners(subgraphs, nb_threads):
	"""
	Create 'nb_threads' DPU runners from 'subgraphs'.
	"""
	all_dpu_runners = []
	for i in range(nb_threads):
		# Create a DpuRunner object from Vitis AI Runtime (VART) 
		all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run")) 
	return all_dpu_runners


def create_threads(all_dpu_runners, nb_threads, num_images, output_folder, output_format, noise):
	"""
	Create a list of 'nb_threads' threads to be run on the DPU, with instance 'all_dpu_runners' 
	to process a total of 'num_images' inputs and place the output in the folder 'output_folder'.
	"""
	# Configure the threads to be run on the DPU
	threadAll = []
	print('Starting', nb_threads, 'thread(s)...')
	# Start index in the list of images to produce
	start_idx = 0
	# Equally divide the number of images to produce by the enumber of threads
	thread_images = num_images // nb_threads
	print('Thread images :', thread_images)
	# Populate thread list
	for i in range(nb_threads):
		# Check whether it is the last thread in the list
		if (i == nb_threads-1):
			end_idx= num_images
		else:
		    	end_idx = start_idx + thread_images
		# Get thread input batch
		in_q = noise[start_idx : end_idx]
		# Instanciate thread
		t1 = threading.Thread(target=runDPU, args=(i, start_idx, all_dpu_runners[i], in_q, output_folder, output_format))
		# Add thread to total list of threads
		threadAll.append(t1)
		# Update start index for next thread
		start_idx = end_idx
	return threadAll


def run_threads(threadAll):
	"""
	Run a list of threads 'threadAll' on the DPU.
	"""
	# Store start time
	#t_begin = time.time()
	# Run the thread(s) on DPU
	for x in threadAll:
		x.start()
	# Wait until the thread(s) terminate(s)
	for x in threadAll:
		x.join()
	# Store end time
	#t_end = time.time()
	# Calculate runtime duration
	return t_end - t_begin


###############################################################################################################

def runDPU(thread_id, start, dpu, inputs, output_folder, output_format): 
	"""
	Thread callback with identifier 'thread_id', to be run on DPU 'dpu', with slice of input data 'inputs' of start index 'start'. 
	The output is stored in folder 'output_folder'.
	"""
	global t_begin
	global t_end

	# Query the DPU runner for the shape and name of the input/output tensors it expects for its loaded Vitis AI model
	inputTensors = dpu.get_input_tensors()
	outputTensors = dpu.get_output_tensors()

	# Get input/output dimensions
	input_ndim = tuple(inputTensors[0].dims)
	output_ndim = tuple(outputTensors[0].dims)

	# Get batch size
	batch_size = inputTensors[0].dims[0]

	# Get number of images to produce
	nb_images = len(inputs)

	img_idx = 0 
	global_idx = start
	while img_idx < nb_images:
		# Check input boundaries 
		if (img_idx + batch_size <= nb_images):
		    	run_size = batch_size
		else:
		    	run_size = nb_images - img_idx

		# Prepare output matrix
		outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

		# Prepare input
		inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]                        
		for j in range(run_size):
			    imageRun = inputData[0]                                                                         
			    imageRun[j,...] = inputs[(img_idx + j) % nb_images].reshape(input_ndim[1:])
            
		# Store start time
		t_begin = time.time()

		# Execute the runner with batch submitting input tensors for execution and output tensors to store results
		job_id = dpu.execute_async(inputData, outputData)
		# Block until the job is complete and the results are ready (wait for the end of DPU processing)
		dpu.wait(job_id)

		# Store end time
		t_end = time.time()

		# Post-processing : convert and save data into output image
		for j in range(run_size):
			# Get output data
			image = outputData[0][j]
			# Change type to uint32
			image = image.astype(np.uint32)
			# Reshape image data
			image = image.reshape(output_height, output_width)
			# Normalize output
			image = image / np.amax(image)
			# Post-process output
			image = image * 255
			# Write image to output folder
			image_name = 'res_' + str(global_idx)
			save_image(image, output_folder, image_name, output_format)
			# Update image index
			global_idx += 1
		# Update index
		img_idx = img_idx + run_size


###############################################################################################################

# Application code to run on the Alveo U280 accelerator card.

###############################################################################################################

import os
import argparse
import glob
import shutil

import numpy as np
import tensorflow as tf

import dpu_runner

###############################################################################################################

# Enable eager execution
tf.enable_eager_execution() 

###############################################################################################################

def app(model, num_threads, output_folder, output_format, num_images, codings_size, seed) :
	"""
	Run the application on the DPU (Deep Learning Processor Unit), a programmable engine dedicated for convolutional neural networks.
	The Xilinx Intermediate Representation (XIR) is a graph-based intermediate representation of the AI algorithms 
	which is designed for compilation and efficient deployment of the DPU on the FPGA platform. 
	XIR includes the Op, Tensor, Graph, and Subgraph libraries, which provide a clear and flexible representation of the computational graph. 
	XIR has in-memory format and file format for different usage. The in-memory format XIR is a graph object and the file format is an xmodel. 
	A graph object can be serialized to an xmodel while an xmodel can be deserialized to a graph object.
	"""

	# Set the seed value
	tf.compat.v1.set_random_seed(seed) 

	# Reset output folder content
	if not os.path.isdir(output_folder):
		# Create output folder
		os.makedirs(output_folder)
		print("Created folder", output_folder)
	else :
		# Delete output folder cosntent
		sub_folders_list = glob.glob(output_folder)
		for sub_folder in sub_folders_list:
	    		shutil.rmtree(sub_folder)
		# Create output folder
		os.makedirs(output_folder)
		print("Folder", output_folder, "already exists. Resetting content...")

	# Create Gaussian noise distribution input 
	noise = tf.random.normal(shape=[num_images, 3, 5, 5])	
	# Convert tensor to numpy
	noise = noise.numpy()

	# Get graph by deserializing the model 
	g = dpu_runner.get_graph(model)

	# Get DPU subgraphs from graph
	subgraphs = dpu_runner.get_subgraph(g)

	# Create DPU runners from the subgraphs
	all_dpu_runners = dpu_runner.create_runners(subgraphs, num_threads)

	# Build thread list
	thread_list = dpu_runner.create_threads(all_dpu_runners, num_threads, num_images, output_folder, output_format, noise)

	# Log output
	print('Generating', num_images, 'images...')

	# Execute the threads
	total_duration = dpu_runner.run_threads(thread_list)

	# Evaluate runtime performances
	print("\nPerformances :")
	dpu_runner.evaluate_perfs(num_images, total_duration)


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-m',  '--model',          type=str,   default='model_dir/dpu_customcnn.xmodel',  help='Path of xmodel. Default is model_dir/dpu_customcnn.xmodel')
	parser.add_argument('-t',  '--threads',        type=int,   default=1,                                 help='Number of threads. Default is 1')
	parser.add_argument('-p',  '--output_folder',  type=str,   default='./output_folder',                 help='Path to output images. Default is ./output_folder.')
	parser.add_argument('-f',  '--output_format',  type=str,   default='png',                             help='Format of output images. Default is png.')
	parser.add_argument('-n',  '--num_images',     type=int,   default=5,                                 help='Number of images to generate. Default is 5.')
	parser.add_argument('-c',  '--codings_size',   type=int,   default=30,                                help='Parameter for Gaussian generation. Default is 30.')
	parser.add_argument('-s',  '--seed',           type=int,   default=42,                                help='Set the seed value for noise generation as input. Default is 42.')

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values   
	print('\n------------------------------------')
	print ('            RUNNING APP')
	print('------------------------------------')
	print ('Command line options:')
	print (' --model : ', 	        args.model)
	print (' --threads : ',         args.threads)
	print (' --output_folder : ',   args.output_folder)
	print (' --output_format : ',   args.output_format)
	print (' --num_images : ',      args.num_images)
	print (' --codings_size : ',    args.codings_size)
	print (' --seed : ',            args.seed)
	print('------------------------------------')

	# Run App
	app(args.model, args.threads, args.output_folder, args.output_format, args.num_images, args.codings_size, args.seed)

if __name__ == '__main__':
	main()


###############################################################################################################

# Run TensorFlow graph of the GAN generator to produce images.

###############################################################################################################

import sys
import os
import argparse
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.decent_q

sys.path.insert(1, './src/file_management')
import file_manager

###############################################################################################################

def post_process(images):
    """
    Invert pixel intensity of 'images' and scale to [0:255].
    """
    return (1 - images) * 255

###############################################################################################################

def inference(graph_name, output_folder, num_images, image_format, seed, input_node, output_node):
	"""
	Run the graph 'graph_name' composed of input node 'input_node' and ouput node 'output_node'.
	Execute the generator to produce 'num_images' images with format 'image_format' in folder 'output_folder'.
	"""
	# Initialize a dataflow graph structure
	input_graph_def = tf.Graph().as_graph_def()
	# Parse the Tensorflow graph
	input_graph_def.ParseFromString(tf.io.gfile.GFile(graph_name, "rb").read())
	# Imports the graph from graph_def into the current default graph
	tf.import_graph_def(input_graph_def, name = '')
	# Get the default graph for the current thread
	graph = tf.compat.v1.get_default_graph()

	# Get input tensor
	noise_in = graph.get_tensor_by_name(input_node+':0')
	# Get shape of input tensor
	input_tensor_size = noise_in.get_shape()[1]
	# Get output tensor
	images_out = graph.get_tensor_by_name(output_node+':0')
	
	# Set seed for random values generation
	tf.compat.v1.set_random_seed(seed)

	# Create input data (noise)     
	noise = tf.random.normal([num_images, input_tensor_size, 1, 1])
	# Convert tensor to numpy array
	noise = noise.eval(session=tf.compat.v1.Session())	

	with tf.compat.v1.Session() as sess:
		# Initializes TensorFlow global variables
		sess.run(tf.compat.v1.global_variables_initializer())
		# Store start time
		t_begin = time.time()
		# Run graph
		raw_images = sess.run(images_out, feed_dict={noise_in: noise})
		# Store end time
		t_end = time.time()

	# Post-process normalized output
	images = post_process(raw_images)
	# Save results to output folder
	file_manager.save_results(images, image_format, output_folder)
	# Compute the duration of the inference process
	return t_end - t_begin


###############################################################################################################

def evaluate_perfs(duration):
    """ 
    Calculate and print the duration of the inference phase.
    """ 
    print("Duration = %.0f ms" %(duration * 1000))


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g',  '--path_to_graph', type=str,   default='./graph.pb',  help="Path to the TensorFlow graph of the GAN's generator. Default is './graph.pb'.")
	parser.add_argument('-o',  '--output_folder', type=str,   default='./out',       help="Folder to store the output images. Default is './out'.")
	parser.add_argument('-n',  '--num_images',    type=int,   default=8,             help="Number of images to produce. Default is '8'.")
	parser.add_argument('-f',  '--format',        type=str,   default='png',         help="Output image format. Default is 'png'.")
	parser.add_argument('-s',  '--seed',          type=int,   default=42,            help="Set seed value. Default is '42'.")
	parser.add_argument('-i',  '--input_node',    type=str,   default='input',       help="Input node name. Default is 'input'.")
	parser.add_argument('-t',  '--output_node',   type=str,   default='output',      help="Output node name. Default is 'output'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --path_to_graph:', args.path_to_graph)
	print(' --output_folder:', args.output_folder)
	print(' --num_images:',    args.num_images)
	print(' --format:',        args.format)
	print(' --seed:',          args.seed)
	print(' --input_node:',    args.input_node)
	print(' --output_node:',   args.output_node)
	print('------------------------------------\n')

	# Run the GAN's generator to produce images
	delta_t = inference(args.path_to_graph, args.output_folder, args.num_images, args.format, args.seed, args.input_node, args.output_node)       

	# Display performances
	print("\nInference performances :")
	evaluate_perfs(delta_t)


###############################################################################################################

if __name__ == '__main__':
	main()

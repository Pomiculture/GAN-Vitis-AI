###############################################################################################################

# Run the Keras model of the GAN generator to produce images.

###############################################################################################################

import sys
import argparse
import time

import tensorflow as tf
from tensorflow import keras
import numpy as np

sys.path.insert(1, './src/file_management')
import file_manager

###############################################################################################################

# Enable TensorFlow eager execution
tf.compat.v1.enable_eager_execution()

###############################################################################################################

def load_keras_model(model_path):
	"""
	Load keras model located in 'model_path'.
	"""
	try:                                          
		# Load Keras model
		model = keras.models.load_model(model_path)
		# Display model summary
		print(model.summary())
	except OSError:
		raise OSError("The path '{0}' does not exist.".format(model_path))
	return model


def generate_images(model, noise):
	"""
	Generate images using the model 'model' from input 'noise'.
	"""
	# Run the GAN's generator
	return model(noise)


def post_process(images):
	"""
	Invert pixel intensity of 'images' and scale to [0:255].
	"""
	return (1 - images.numpy()) * 255


###############################################################################################################

def inference(path_to_model, output_folder, num_images, format, seed):
	"""
	Execute the generator 'path_to_model' to produce 'num_images' images with format 'format' in folder 'output_folder'.
	"""
	# Set seed in order to execution stable across runs
	tf.compat.v1.set_random_seed(seed)

	# Load Keras model
	generator = load_keras_model(path_to_model)
	# Get model input layer's length
	input_length = generator.layers[0].input_shape[1]
	print('Input length', input_length)
	# Create input data (Gaussian distribution - white noise)
	noise = tf.random.normal([num_images, input_length, 1, 1])

	# Store start time
	t_begin = time.time()
	# Run the model to produce images
	images = generate_images(generator, noise)
	# Store end time
	t_end = time.time()

	# Process images
	images = post_process(images)
	# Store results in output folder
	file_manager.save_results(images, format, output_folder)
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
	parser.add_argument('-p',  '--path_to_model', type=str,   default='./keras_h5/keras_generator_model.h5',  help="Path to Keras model of the GAN's generator. Default is './keras_h5/keras_generator_model.h5'.")
	parser.add_argument('-o',  '--output_folder', type=str,   default='./out',                                help="Folder to store the output images. Default is './out'.")
	parser.add_argument('-n',  '--num_images',    type=int,   default=8,                                      help="Number of images to produce. Default is '8'.")
	parser.add_argument('-f',  '--format',        type=str,   default='png',                                  help="Output image format. Default is 'png'.")
	parser.add_argument('-s',  '--seed',          type=int,   default=42,                                     help="Set seed value. Default is '42'.")

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --path_to_model:', args.path_to_model)
	print(' --output_folder:', args.output_folder)
	print(' --num_images:',    args.num_images)
	print(' --format:',        args.format)
	print(' --seed:',          args.seed)
	print('------------------------------------\n')

	# Run the GAN's generator to produce images
	delta_t = inference(args.path_to_model, args.output_folder, args.num_images, args.format, args.seed)       
	
	# Display performances
	print("\nInference performances :")
	evaluate_perfs(delta_t)

###############################################################################################################

if __name__ == '__main__':
	main()

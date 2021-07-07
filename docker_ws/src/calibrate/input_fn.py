###############################################################################################################

# Calibrate graph for quantization step

###############################################################################################################

import os
import tensorflow as tf
import numpy as np

###############################################################################################################

# Get the environment variables (number of images in the batch, input resolution)
num_batch_images = int(os.environ['CALIB_BATCH_SIZE']) 	
codings_size = int(os.environ['CODINGS_SIZE']) 
num_iter = int(os.environ['NB_ITER']) 
input_tensor = os.environ['INPUT_NODE_NAME']

# Log total number of images to process
print("Processing {} images...".format(num_iter * num_batch_images))

def calib_input(iter):
	'''
	Input of the GAN generator algorithm for calibration during the quantization process.
	'''
	# Set seed for random values generation
	tf.compat.v1.set_random_seed(iter)
	# Generate noisy input of size 'num_batch_images' images
	noise = tf.random.normal([num_batch_images, codings_size, 1, 1])
	# Convert tensor to numpy array
	noise = noise.eval(session=tf.compat.v1.Session())	
	# Link input noise to input node name
	return {input_tensor : noise}

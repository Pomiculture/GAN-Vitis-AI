###############################################################################################################

# Build and train GAN model using TensorFlow-Keras framework.

"""
Source : https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb
"""

###############################################################################################################

import sys
import os
import argparse

import tensorflow as tf
import numpy as np

###############################################################################################################

# Trigger TensorFlow eager execution mode
tf.compat.v1.enable_eager_execution()

###############################################################################################################

def build_gan_layers(codings_size, output_width, output_height):
	"""
	Build GAN generator and discriminator.
	"""
	# Create generator                                        
	generator = tf.keras.Sequential([
		tf.keras.layers.Reshape([codings_size], input_shape=(codings_size, 1 ,1)),
		tf.keras.layers.Dense(100, activation="relu"), 
		tf.keras.layers.Dense(150, activation="relu"),
		tf.keras.layers.Dense(output_width * output_height, activation="sigmoid"),
		tf.keras.layers.Reshape([output_width, output_height])],
		name='Generator')	

	# Create discriminator (binary classificator : "0" for generated pictures & "1" for real ones)
	discriminator = tf.keras.Sequential([
		tf.keras.layers.Flatten(input_shape=[output_width, output_height]),
		tf.keras.layers.Dense(150, activation="selu"),
		tf.keras.layers.Dense(100, activation="selu"),
		tf.keras.layers.Dense(1, activation="sigmoid")], 
		name='Discriminator')

	# Display model's characteristics
	print(generator.summary())
	print(discriminator.summary())

	return generator, discriminator


###############################################################################################################

def train_discriminator(batch_size, codings_size, X_batch, generator, discriminator):
	"""
	Train GAN disciminator.
	"""
	noise = tf.random.normal(shape=[batch_size, codings_size, 1, 1])
	generated_images = generator(noise)
	X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
	y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
	discriminator.trainable = True
	discriminator.train_on_batch(X_fake_and_real, y1)
	return generated_images


def train_generator(batch_size, codings_size, gan, discriminator):
	"""
	Train GAN generator.
	"""
	noise = tf.random.normal(shape=[batch_size, codings_size, 1, 1])
	y2 = tf.constant([[1.]] * batch_size)
	discriminator.trainable = False
	gan.train_on_batch(noise, y2)                    


def train_gan(gan, dataset, batch_size, codings_size, n_epochs=1):
	"""
	Train GAN network (generator & discriminator).
	"""
	# Extract GAN submodels
	generator, discriminator = gan.layers
    
	for epoch in range(n_epochs):

        	print("Epoch {}/{}".format(epoch + 1, n_epochs))             

        	for X_batch in dataset:
            		# phase 1 - training the discriminator
            		generated_images = train_discriminator(batch_size, codings_size, X_batch, generator, discriminator)
            		# phase 2 - training the generator
            		train_generator(batch_size, codings_size, gan, discriminator)


###############################################################################################################

def prepare_data(batch_size):
	"""
	Get data from Fashion MNIST dataset and process it.
	"""
	# Load Fasion MNIST dataset
	print('Loading dataset...')
	(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

	# Normalize training set
	rgb_size = 255
	X_train_full = X_train_full.astype(np.float32) / rgb_size

	# Normalize testing set
	#X_test = X_test.astype(np.float32) / rgb_size

	# Split dataset into training and validation data 
	split_value = 5000
	X_train, X_valid = X_train_full[:-split_value], X_train_full[-split_value:]
	y_train, y_valid = y_train_full[:-split_value], y_train_full[-split_value:]

	# Configure dataset
	dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
	dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
	return dataset


def save_gan_layers(generator, discriminator, folder_name, generator_filename, discriminator_filename):
	"""
	Save weights, model architecture & optimizer to an HDF5 format file.
	"""
 	# Save generator
	generator.save(os.path.join(folder_name,generator_filename))             
	# Save discriminator     
	discriminator.save(os.path.join(folder_name,discriminator_filename))
	print("Files {0} and {1} saved to path {2}".format(generator_filename, discriminator_filename, folder_name))


###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-c',  '--codings_size',   		type=int,   	default=75,        		help="Set parameter for Gaussian distribution number of elements. Default is '75'.")
	parser.add_argument('-s',  '--seed',           		type=int,   	default=42,             	help="Set seed value. Default is '42'.")
	parser.add_argument('-b',  '--batch_size',     		type=int,   	default=32,             	help="Set batch size for training. Default is '32'.")
	parser.add_argument('-e',  '--epochs',         		type=int,   	default=1,             		help="Set number of epochs for training. Default is '1'.")
	parser.add_argument('-w',  '--output_width',   		type=int,   	default=5,              	help="Set image output width. Default is '5'.")
	parser.add_argument('-t',  '--output_height',  		type=int,   	default=5,               	help="Set image output height. Default is '5'.")
	parser.add_argument('-o',  '--output_folder',  		type=str,   	default='./train',  		help="Path to output Keras saved models. Default is './train'.")
	parser.add_argument('-g',  '--generator_name', 		type=str,   	default='generator.h5',  	help="Name of generator saved model. Default is 'generator.h5'.")
	parser.add_argument('-d',  '--discriminator_name', 	type=str,   	default='discriminator.h5',  	help="Name of discriminator saved model. Default is 'discriminator.h5'.")
	# Parse arguments
	args = parser.parse_args()  

	# Display Python-TF-Keras versions
	print('\n------------------------------------')
	print('Python version     :', (sys.version))
	print('TensorFlow version :', tf.__version__)
	print('Keras version      :', tf.keras.__version__)

	# Print argument values
	print('------------------------------------')
	print ('Command line options:')
	print(' --seed:',           	args.seed)
	print(' --codings_size:',   	args.codings_size)	
	print(' --epochs:',         	args.epochs)
	print(' --batch_size:',     	args.batch_size)	
	print(' --output_width:',   	args.output_width)
	print(' --output_height:',  	args.output_height)
	print(' --output_folder:',  	args.output_folder)
	print(' --generator_name:',  	args.generator_name)
	print(' --discriminator_name:', args.discriminator_name)
	print('------------------------------------\n')

	# Fix random seed                                            
	tf.compat.v1.set_random_seed(args.seed)

	# Build GAN generator and discriminator
	generator, discriminator = build_gan_layers(args.codings_size, args.output_width, args.output_height)

	# Assemble GAN network from both generator and discriminator
	gan = tf.keras.Sequential([generator, discriminator])

	# Compile discriminator, loss : binary cross-entropy
	discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
	discriminator.trainable = False

	# Compile GAN, loss : binary cross-entropy
	gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

	# Prepare dataset for training
	dataset = prepare_data(args.batch_size)

	# Training phase
	train_gan(gan, dataset, args.batch_size, args.codings_size, n_epochs=args.epochs)

	# Save weights, model architecture & optimizer to an HDF5 format file
	save_gan_layers(generator, discriminator, args.output_folder, args.generator_name, args.discriminator_name)


###############################################################################################################

if __name__ == '__main__':
	main()

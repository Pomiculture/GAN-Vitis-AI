import sys
import argparse
import tensorflow as tf
import numpy as np

sys.path.insert(1, './src/file_management')
import file_manager

###############################################################################################################
  
def download_images(output_folder, nb_images, img_format):
	'''
	Download 'nb_images' from the Fashion MNIST dataset and export them to the folder 'output_folder' with format 'img_format'.
	'''

	# Extract training set from Fashion MNIST dataset
	(X_train_full, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

	# Set the data type
	X_train_full = X_train_full.astype(np.float32)

	# Select a specific number of images
	X_train = X_train_full[:nb_images]

	# Create the output folder
	file_manager.create_empty_folder(output_folder)

	# Save the images in the folder
	file_manager.save_results(X_train, img_format, output_folder)

###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-o',  '--output_folder', type=str,   default='./out',   help="Output folder containing the downloaded content. Default is './out'.")
	parser.add_argument('-n',  '--nb_images',     type=int,   default=50,        help="Number of images to download. Default is '50'.")
	parser.add_argument('-f',  '--img_format',    type=str,  default='png',      help="Format of the output images. Default is 'png'.")
		                    
	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --output_folder:', args.output_folder)
	print(' --nb_images:',     args.nb_images)
	print(' --img_format:',    args.img_format)
	print('------------------------------------\n')

	# Download images from the dataset
	download_images(args.output_folder, args.nb_images, args.img_format)


###############################################################################################################

if __name__ == '__main__':
    main()

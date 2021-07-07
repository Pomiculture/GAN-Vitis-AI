import os
import sys
import argparse

import cv2
import numpy as np
from tensorflow import keras

sys.path.insert(1, './src/file_management')
import file_manager

###############################################################################################################

def pre_process(image):
    """
    Invert pixel intensity of 'images' (to compensate the conversion into image with imwrite).
    """
    return 1 - image * 255


###############################################################################################################
#     
def run_discriminator(folder, model_path):
    '''
    Run the discriminator 'model_path' over the images from the folder 'folder'. Display the mean, min and max values, and percentage of good predictions.
    '''
    # Get images from folder
    file_names = file_manager.get_content_from_folder(folder)

    # Load Keras model
    model = keras.models.load_model(model_path)

    # Display model summary
    print(model.summary())
    
    # Process images
    predictions = []
    for name in file_names:
        # Extract (grayscale) image from file 
        raw_img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
        # Preprocess image
        img = pre_process(raw_img) 
        # Extend image dimensions
        img_extended = np.expand_dims(img, axis=0)
        # Run prediction
        score = model.predict(img_extended)
        # Append score to list
        predictions.append(float(score))
    
    # Display mean, min, max from the results
    print('Results over', len(file_names), 'images.')
    print('Average score:', np.round(np.mean(predictions), 3), '- Min:', np.round(min(predictions), 3), '- Max:', np.round(max(predictions),3))
    # Get number of good predictions
    num_success = len([p for p in predictions if p > 0.5])

    # Percentage of images above 50%
    print("Percentage of images above 50%: {0}%".format(num_success / len(predictions) * 100))
    
    return predictions


###############################################################################################################

def main():

    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',  '--folder',  type=str,   default='./out',                                    help="Folder to read image data from. Default is './out'.")
    parser.add_argument('-m',  '--model',   type=str,   default='./keras_h5/keras_discriminator_model.h5',  help="Path to the Keras GAN's discriminator model. Default is './keras_h5/keras_discriminator_model.h5'.")
    
    # Parse arguments
    args = parser.parse_args()  

    # Print argument values
    print('\n------------------------------------')
    print ('Command line options:')
    print(' --folder:', args.folder)
    print(' --model:',  args.model)
    print('------------------------------------\n')

    # Run the discriminator over the images to evaluate their degree of realism
    run_discriminator(args.folder, args.model)


###############################################################################################################

if __name__ == '__main__':
    main()

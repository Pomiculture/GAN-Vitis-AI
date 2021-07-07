import os

# Disable Tensorflow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#############################################################################################################

import numpy as np
from scipy.linalg import sqrtm

from keras.applications.inception_v3 import preprocess_input  
from keras.applications.inception_v3 import InceptionV3
from skimage.transform import resize

#############################################################################################################

# scale an array of images to a new size
def scale_images(images, new_shape):
    """
    Resize the arrays of pixel values 'images' to the required shape 'new_shape'.
    """
    images_list = list()
    for image in images:
        # Resize image with Nearest-neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # Store modified image
        images_list.append(new_image)
    return np.asarray(images_list)


def preprocess_images(images, name, shape):
    """
    Format image data 'images' from batch 'name' to be the input of the InceptionV3 model, whose input layer has a shape of 'shape'.
    """
    # Convert list to array with floating point values
    #images = np.array(images, dtype=np.float64)
    images = np.array(images) 
    print('Shape of raw batch of', name, 'images :', images.shape) 
    # Resize images
    images = scale_images(images, shape)
    print('Shape of scaled batch of', name, 'images :', images.shape, '\n')                      
    # Scale images (the inputs pixel values are scaled between -1 and 1 sample-wise) 
    return preprocess_input(images)


###############################################################################################################

def get_gaussian_parameters(distribution):
    """
    Return the mean and covariance from the multivariate normal distribution 'distribution'.
    """
    return np.mean(distribution, axis=0), np.cov(distribution, rowvar=False)


def ssdiff(mu_1, mu_2):
    """
    Calculate the sum squared difference between the two mean vectors 'mu_1' and 'mu_2'.
    """
    return np.sum((mu_1 - mu_2)**2.0)


def covmean(sigma_1, sigma_2):
    """
    Calculate the square root of the product between covariances 'sigma_1' and 'sigma_2'.
    """
    # Calculate square root of product between covariances
    covmean = sqrtm(sigma_1.dot(sigma_2))
    # Remove imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return covmean

    
def compute_fid(model, images1, images2):
    """
    Calculate the Frechet Inception Distance d².
    The formula is : d² = ||mu_1 - mu_2||² + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    With 'mu' the feature-wise mean of the real and generated images,
    and 'C' the covariance matrix (sigma) for the real and generated feature vectors.
    Each image is predicted by the InceptionV3 model as 2,048 activation features.
    """
	# Calculate activations
    activations_1 = model.predict(images1)
    activations_2 = model.predict(images2)

    # Calculate mean and standard deviation statistics 
    mu_1, sigma_1 = get_gaussian_parameters(activations_1)           
    mu_2, sigma_2 = get_gaussian_parameters(activations_2)    

    # Calculate the FID score
    return ssdiff(mu_1, mu_2) + np.trace(sigma_1 + sigma_2 - 2.0 * covmean(sigma_1, sigma_2))


###############################################################################################################

def fid(output_images, gold_images):
    """
    Calculate the Frechet Inception Distance (FID) between two batches of images : 'output_images' and 'gold_images'.
    This metric is used to evaluate GAN's output quality by measuring the distance between feature vectors calculated for real and generated images.
    The compared feature vectors come from the Inception V3 classificator, pre-trained on the ImageNet dataset.
    The goal is to evaluate synthetic images based on the statistics of a collection of synthetic images compared to the statistics of a collection of real images from the target domain.
    A score of 0.0 indicates that the batch images have the exact same content.

    InceptionV3 model :
    - The input shape has to be (299, 299, 3);
    - Global average pooling is applied to the output of the last convolutional block -> the output of the model is a 2D tensor;
    - The weights are obtained from pre-training on ImageNet dataset.
    """
    # InceptionV3 input shape
    input_shape = (299, 299, 3)

    # Preprocess images to apply InceptionV3 model
    output_images = preprocess_images(output_images, 'output', input_shape)                              
    gold_images = preprocess_images(gold_images, 'gold', input_shape)
 
    # Load the InceptionV3 model 
    model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape, weights='imagenet') 

    # Calculate FID
    fid = compute_fid(model, output_images, gold_images)
    print('FID: %.2f' % fid)
    return True


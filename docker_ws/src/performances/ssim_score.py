import cv2
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from statistics import mean

###############################################################################################################

def display_image(img):
    """
    Display the image 'img'.
    """
    plt.imshow(img, cmap="binary")
    plt.axis("off")
    plt.show()


def compute_ssim(img_1, img_2):
    """
    Compute the Structural Similarity Index (SSIM) between the two images 'img_1' and 'img_2'.
    """
    # Calculate the similarity between both images
    (score, diff_img) = structural_similarity(img_1, img_2, full=True)
    # Get the difference image
    diff_img = (diff_img * 255).astype("uint8")
    # Display the difference image
    #display_image(diff_img)
    # Log and return the SSIM score
    #print("SSIM: {}".format(round(score, 2)))
    return score


def similarity_verdict(score):
    """
    Indicate the quality of the similarity score 'score'.
    """
    print("=> High score") if (score > 0.8) else print("=> Low score")
    return


###############################################################################################################

def ssim(output_images, gold_images):
    '''
    Evaluate the similarity between the output images 'output_images'
    and the ones produced by the reference 'gold_images',
    one against one using the Structural Similarity Index (SSIM).
    The score is 1.0 if the batch images have the exact same content.
    '''

    # Run through folders's elements to compare the respective pairs of images
    ssim_all = []
    for i in range(len(output_images)) :
        # Read respective image from both folders
        img_output = output_images[i]
        img_gold = gold_images[i]

        # Check whether the dimensions match
        assert img_gold.shape == img_output.shape, "ERROR : Images's size don't match. Stopping SSIM comparison ...\n"    

        # Convert images to grayscale
        gray_img_out = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
        gray_img_gold = cv2.cvtColor(img_gold, cv2.COLOR_BGR2GRAY)

        # Apply SSIM (Structural Similarity Index)
        ssim = compute_ssim(gray_img_out, gray_img_gold)

        # Append SSIM score to the total list
        ssim_all.append(ssim)

    # Calculate the average SSIM score between the output images and the gold ones
    ssim_mean = mean(ssim_all)

    # Print the average value over the SSIM scores
    print("Average SSIM score between output and gold over the", len(output_images), "images:", round(ssim_mean, 2))
    # Indicate the quality of similarity between both folders's content
    similarity_verdict(ssim_mean)
    print("Percentage : ",  round(ssim_mean*100, 2), "%")
    return True


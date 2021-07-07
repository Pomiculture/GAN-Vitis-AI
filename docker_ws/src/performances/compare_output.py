import sys
import argparse
from enum import Enum

import ssim_score
import fid_score

sys.path.insert(1, './src/file_management')
import file_manager

###############################################################################################################

class Mode(Enum):
    SSIM = 'ssim'
    FID = 'fid'

    def __str__(self):
        return self.value

###############################################################################################################
  
def evaluate_accuracy(output_folder, gold_folder, mode):
    '''
    Evaluate the similarity between the output images, from 'output_folder' 
    and the ones produced by the reference (gold), from 'gold_folder' :
    - One against one (gold data is the one produced by the original GAN) using the Structural Similarity Index (SSIM);
    - Between both batches (gold data is composed of real images) with the Frechet Inception Distance (FID).
    '''
    # Read folders's content
    output_content = file_manager.get_content_from_folder(output_folder)
    gold_content = file_manager.get_content_from_folder(gold_folder)

    # Get the number of images to process
    nb_elements = len(output_content)
    if(nb_elements <= 1):
            print("ERROR : Not enough images to compare. Exiting program ...")
            return False

    # Check content size
    if(mode == Mode.SSIM):
        print('Calculating SSIM score...')
        # Calculate size content of folders to check if they match each other
        if(nb_elements != len(gold_content)):
            print("ERROR : Folders's size don't match. Exiting program ...")
            return False
    elif(mode == Mode.FID):
        print('Calculating FID score...')
        # Check whether there is enough real images 
        if(len(gold_content) >= nb_elements):
            # Match the number of real images with the number of generated ones
            gold_content = gold_content[:nb_elements]
        else:
            print("ERROR : Not enough reference images. Exiting program ...")
            return False   
        

    # Get the images from the content
    output_images = file_manager.get_images_from_files(output_folder, output_content)
    gold_images = file_manager.get_images_from_files(gold_folder, gold_content)
    
    # Act according to the chosen mode
    if(mode == Mode.SSIM):
        # Calculate SSIM score
        ssim_score.ssim(output_images, gold_images)    
    elif(mode == Mode.FID):
        # Calculate FID score
        fid_score.fid(output_images, gold_images)       

    return True


###############################################################################################################

def main():

    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',  '--output_folder',   type=str,   default='./out',   help="Output folder's name, containing the images produced by the GAN. Default is './out'.")
    parser.add_argument('-g',  '--gold_folder',     type=str,   default='./gold',  help="Gold folder's name, containing reference images. Default is './gold'.")
    parser.add_argument('-m',  '--mode',            type=Mode,  default='fid',    help="Metric used for comparison. Default is 'fid'.", choices=list(Mode))
                                    
    # Parse arguments
    args = parser.parse_args()  

    # Print argument values
    print('\n------------------------------------')
    print ('Command line options:')
    print(' --output_folder:', args.output_folder)
    print(' --gold_folder:',   args.gold_folder)
    print(' --mode:',          args.mode)
    print('------------------------------------\n')

    # Evaluate difference between the corresponding pairs of images
    evaluate_accuracy(args.output_folder, args.gold_folder, args.mode)


###############################################################################################################

if __name__ == '__main__':
    main()

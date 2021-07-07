import os
import sys
import argparse
import random

from pydrive2.drive import GoogleDrive
from pydrive2.files import GoogleDriveFileList

import google_drive_utils

sys.path.insert(1, './src/file_management')
import file_manager

###############################################################################################################

def find_folder(folder_list, folder_name):
    """
    Check whether there is already a folder named 'folder_name' within the list 'folder_list'. Return its id.
    """
    # Check whether a folder with the same name already exists 
    for folder in folder_list:
        if folder['title'] == folder_name:
            #print('title: %s, id: %s' % (folder['title'], folder['id']))
            return folder['id']

    return False


def download_files(drive, content, local_folder, nb_imgs):
    """
    Download 'nb_imgs' images from 'drive' folder 'content' to target 'local_folder'.
    """
    # Shuffle the order of the Google Drive file list
    shuffled_content = random.sample(content, len(content))     

    # Download the files
    i = 0
    for file in shuffled_content:
        # Download the image file
        img = drive.CreateFile({'id': file['id']})
        img.GetContentFile(os.path.join(local_folder, file['title'])) 
        print('Downloaded image', file['title'], 'to local folder', local_folder)
        # Increment the file count
        i += 1
        # Check whether the requested number of images has been downloaded
        if(i == nb_imgs):
            print('Downloaded the', i, 'images.')
            break
    # Check whether the requested number of images has been downloaded
    if(i < nb_imgs):
        print('Not enough images in folder. Downloaded', i, 'images instead of the', nb_imgs, 'requested.')

###############################################################################################################

def download_from_google_drive(credentials, parent_drive_id, drive_folder_name, local_folder, nb_images):
    """
    Upload 'local_folder' content to a folder on Google Drive 'drive_folder_name'.
    """
    # Google authentication 
    g_auth = google_drive_utils.authenticate(credentials)

    # Create Google Drive instance using authenticated GoogleAuth instance
    drive = GoogleDrive(g_auth)

    # Check whether the parent Google Drive folder exists
    folder_list = google_drive_utils.get_drive_folder_content(drive, parent_drive_id)
    if not folder_list:
        return False

    # Check whether the Google Drive subfolder exists
    drive_folder_id = find_folder(folder_list, drive_folder_name)
    if not drive_folder_id:
        return False

    # Create the local folder (reset content if already exists)
    file_manager.create_empty_folder(local_folder)

    #Get content from Google Drive folder
    content = google_drive_utils.get_drive_folder_content(drive, drive_folder_id)
    if not content:
        return False

    # Download files from Google Drive folder to the local folder
    download_files(drive, content, local_folder, nb_images)
    
    return True


###############################################################################################################

def main():

    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',  '--credentials',       type=str,   default='client_secrets.json',                                   help='NPath to the client credientials for the Google Drive application. Default is client_secrets.json.')
    parser.add_argument('-p',  '--parent_drive_id', type=str,   default='1peH5eJvFPtHqbAOO2cIlG8kn4zLquBIw', help='ID of parent Google Drive folder. Default is 1peH5eJvFPtHqbAOO2cIlG8kn4zLquBIw.')
    parser.add_argument('-d',  '--drive_folder',    type=str,   default='dataset',                           help='Google Drive folder from which we want to download the files. Default is dataset.')
    parser.add_argument('-l',  '--local_folder',    type=str,   default='input',                             help='Local folder''s name we want to create. Default is input.')
    parser.add_argument('-n',  '--nb_images',       type=int,   default=4,                                   help='Number of images to download from Google Drive folder. Default is 4.')
    
    # Parse arguments
    args = parser.parse_args()  

    # Print argument values
    print('\n------------------------------------')
    print ('Command line options:')
    print(' --credentials:',     args.credentials)
    print(' --parent_drive_id:', args.parent_drive_id)
    print(' --drive_folder:',    args.drive_folder)
    print(' --local_folder:',    args.local_folder)
    print(' --nb_images:',       args.nb_images)
    print('------------------------------------\n')

    # Download the Google Drive folder's content to a new local folder
    download_from_google_drive(args.credentials, args.parent_drive_id, args.drive_folder, args.local_folder, args.nb_images)


###############################################################################################################

if __name__ == '__main__':
    main()

import os
import sys
import argparse

from pydrive2.drive import GoogleDrive

import google_drive_utils

sys.path.insert(1, './src/file_management')
import file_manager

###############################################################################################################

def get_folder_offset(folder_list, folder_name):
    """
    Check whether there is already a folder named 'folder_name' and if so, increment a counter.
    """
    isExists = False

    # Check whether a folder with the same name already exists 
    for folder in folder_list:
        if folder['title'] == folder_name:
            #print('title: %s, id: %s' % (folder['title'], folder['id']))
            isExists = True
            print('Folder', folder_name, 'already exists. Adding an index...')
            break
    if not isExists :
        return 0

    # Increment a counter until finding a folder name that doesn't exist (with pattern 'name_i')
    i=1
    while 1 :
        isNew = True
        for folder in folder_list:
            if folder['title'] == folder_name + '_' + str(i) :
                #print('title: %s, id: %s' % (folder['title'], folder['id']))
                isNew = False
                # Increment the counter
                i+=1
                # Move on to next iteration
                continue
        if(isNew):
            break
    return i


def create_drive_folder(drive, parent_drive_id, folder_name):
    """
    Create a new folder with name 'folder_name' on Google Drive of instance 'drive'.
    """
    folder = drive.CreateFile({'title': folder_name, "mimeType": "application/vnd.google-apps.folder", 'parents': [{'id': parent_drive_id}]})
    folder.Upload()
    print("Folder", folder_name, "successfully created on Google Drive with id :", folder['id'])
    return folder


def upload_files(drive, drive_folder, local_folder, files):
    """
    Upload 'files' of 'local_folder' to Google Drive folder 'drive_folder' of instance 'drive'.
    """
    for image_name in files:
        # Create GoogleDriveFile instance
        img = drive.CreateFile({'parents': [{'id': drive_folder['id']}]})
        # Set file content
        img.SetContentFile(os.path.join(local_folder, image_name))
        # Rename file
        img['title'] = image_name
        # Upload the file to the destination folder
        img.Upload() 
        print("Image", image_name, "uploaded with name", img['title'], "and id", img['id'])


###############################################################################################################

def upload_to_google_drive(credentials, local_folder, parent_drive_id, drive_folder_name):
    """
    Upload 'local_folder' content to a folder on Google Drive 'drive_folder_name'.
    """
    # Google authentication 
    g_auth = google_drive_utils.authenticate(credentials)

    # Create Google Drive instance using authenticated GoogleAuth instance
    drive = GoogleDrive(g_auth)

    #Check whether the parent folder exists
    folder_list = google_drive_utils.get_drive_folder_content(drive, parent_drive_id)
    if not folder_list :
        return False

    # Check whether folder already exists
    offset = get_folder_offset(folder_list, drive_folder_name)
    if(offset > 0):
        drive_folder_name = drive_folder_name + '_' + str(offset)

    # Create the output folder on the Drive space
    drive_folder = create_drive_folder(drive, parent_drive_id, drive_folder_name) 

    # Read local folder's content
    local_content = file_manager.get_content_from_folder(local_folder)
    if(len(local_content) == 0) :
        print('No elements were found in the folder', local_folder)
        return False

    # Upload images to Google Drive folder
    upload_files(drive, drive_folder, local_folder, local_content)
    
    return True


###############################################################################################################

def main():

    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',  '--credentials',       type=str,   default='client_secrets.json',                                   help='NPath to the client credientials for the Google Drive application. Default is client_secrets.json.')
    parser.add_argument('-l',  '--local_folder',    type=str,   default='output',                             help='Local folder''s name, containing the images produced by the GAN. Default is output.')
    parser.add_argument('-p',  '--parent_drive_id', type=str,   default='1peH5eJvFPtHqbAOO2cIlG8kn4zLquBIw',  help='ID of parent Google Drive folder. Default is 1peH5eJvFPtHqbAOO2cIlG8kn4zLquBIw.')
    parser.add_argument('-d',  '--drive_folder',    type=str,   default='dataset',                            help='Folder''s name to create on Google Drive, which will contain the content of the local folder. Default is dataset.')

    # Parse arguments
    args = parser.parse_args()  

    # Print argument values
    print('\n------------------------------------')
    print ('Command line options:')
    print(' --credentials:',     args.credentials)
    print(' --local_folder:',    args.local_folder)
    print(' --parent_drive_id:', args.parent_drive_id)
    print(' --drive_folder:',    args.drive_folder)
    print('------------------------------------\n')

    # Upload the local folder's content to GoogleDrive
    upload_to_google_drive(args.credentials, args.local_folder, args.parent_drive_id, args.drive_folder)
    print('Upload complete.')


###############################################################################################################

if __name__ == '__main__':
    main()

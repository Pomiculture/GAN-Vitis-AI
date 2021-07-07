import os
import glob
import shutil
import cv2

###############################################################################################################

def create_empty_folder(folder_name):
	""" 
	Create new output folder with name 'folder_name'.
	"""
	if not os.path.isdir(folder_name):
		# Create output folder
		os.makedirs(folder_name)
		print("Created folder", folder_name)
	else :
		# Delete output folder content
		sub_folders_list = glob.glob(folder_name)
		for sub_folder in sub_folders_list:
			shutil.rmtree(sub_folder)
		# Create output folder
		os.makedirs(folder_name)
	print("Folder", folder_name, "already exists. Resetting content...")
    

def get_content_from_folder(folder_name):
	"""
	Get the files from the folder named 'folder_name'.
	"""
	try:
		content = os.listdir(folder_name)
		return content
	except FileNotFoundError:
		raise FileNotFoundError("The path {0} doesn't exist.".format(folder_name)) 
  

def get_images_from_files(folder_name, content):
	"""
	Convert the 'content' files from folder 'folder_name' to images.
	"""
	images = []
	for i in range(len(content)) :
		# Read respective image from both folders
		image = cv2.imread(os.path.join(folder_name, content[i]))
		# Append image to list
		images.append(image)
	return images


###############################################################################################################
  
def save_image(image, path, file_name, format) :
	""" 
	Save image 'image' as file 'file_name' with format 'format' in folder 'path'.
	"""  
	print('------------------------------------')
	try:
		iret = cv2.imwrite(os.path.join(path, file_name + '.' + format), image)
		print ('Image', file_name , 'successfully saved in folder', path)   
	except:
		print('ERROR : Failed to save', file_name)   


def save_results(images, format, folder):
	"""
	Save model output 'images' in folder 'folder' with format 'format'.
	"""
	# Create/reset output folder
	create_empty_folder(folder)
	# Write out images
	i = 1
	for image in images:
		# Save image
		image_name = 'res_' + str(i)
		save_image(image, folder, image_name, format)
		# Update image index
		i+= 1


def display_image(image, window_name) :
	""" 
	Display 'image' in a window named 'window_name'.
	"""
	# Display window
	cv2.imshow(window_name, image)
	# Wait for user to press any key
	cv2.waitKey(0)
	# Destroy window
	cv2.destroyWindow(window_name)

    

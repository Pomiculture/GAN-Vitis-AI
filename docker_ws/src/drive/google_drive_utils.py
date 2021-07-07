from pydrive2.auth import GoogleAuth
from pydrive2.files import GoogleDriveFileList, ApiRequestError

###############################################################################################################

def authenticate(client_json_path):
	"""
	Google authentication (requires 'client_secrets.json' credentials file).
	"""
	GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = client_json_path
	google_auth = GoogleAuth()
	#try:
	google_auth.LocalWebserverAuth()
		#print(ret)
	#except KeyboardInterrupt:
	#	print('Google Drive authentification complete.')
		
	return google_auth	


###############################################################################################################

def get_drive_folder_content(drive, id):
	"""
	Check whether the folder with id 'id' exists in Google Drive 'drive' and get the list of sub-folders.
	"""
	# Auto-iterate through all files in the parent folder
	file_list = GoogleDriveFileList()
	try:
		file_list = drive.ListFile({'q': "'{0}' in parents and trashed=false".format(id)}).GetList()
		if len(file_list) == 0:
			print('No file found in Google Drive folder with id', id)
			return False
	except ApiRequestError:
		# Exit if the parent folder doesn't exist
		print('Folder not found with id :', id)
		return False

	return file_list

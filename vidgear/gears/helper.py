"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
"""

# Contains all the support functions/modules required by Vidgear 

# import the necessary packages
import os, sys
import cv2
import numpy as np
from pkg_resources import parse_version
import logging as log
import logging.config

#logging formatter
logging.config.dictConfig({
	'version': 1,
	'formatters': {
		'colored': {
			'()': 'colorlog.ColoredFormatter',
			'log_colors': {
						'DEBUG':    'bold_green',
						'WARNING':  'bold_yellow',
						'ERROR':    'bold_red',
						'CRITICAL': 'bold_red,bg_white',
					},
			'format':
				"%(bold_blue)s%(name)s%(reset)s :: %(log_color)s%(levelname)s%(reset)s :: %(message)s",
		}
	},
	'handlers': {
		'stream': {
			'class': 'logging.StreamHandler',
			'formatter': 'colored',
			'level': 'DEBUG'
		},
	},
	'loggers': {
		'': {
			'handlers': ['stream'],
			'level': 'DEBUG',
		},
	},
})
#logger
logger = log.getLogger('Helper')


def check_CV_version():
	"""
	returns current OpenCV binary version's first bit 
	"""
	if parse_version(cv2.__version__) >= parse_version('4'):
		return 4
	else:
		return 3



def capPropId(property):
	"""
	Retrieves the OpenCV property Integer(Actual) value. 
	"""
	integer_value = 0 
	try:
		integer_value = getattr(cv2, property)
	except Exception:
		logger.critical('{} is not a valid OpenCV property!'.format(property))
		return None
	return integer_value



def dict2Args(param_dict):
	"""
	converts dict to list(args)
	"""
	args = []
	for key in param_dict.keys():
		args.append(key)
		args.append(param_dict[key])
	return args



def get_valid_ffmpeg_path(custom_ffmpeg = '', is_windows = False, ffmpeg_download_path = '', logging = False):
	"""
	Validate the FFmpeg path/binaries and returns valid FFmpeg file executable location(also downloads static binaries on windows) 
	"""
	final_path = ''
	if is_windows:
		#checks if current os is windows
		if custom_ffmpeg:
			#if custom FFmpeg path is given assign to local variable
			final_path += custom_ffmpeg
		else:
			#otherwise auto-download them
			try:
				if not(ffmpeg_download_path):
					#otherwise save to Temp Directory
					import tempfile
					ffmpeg_download_path = tempfile.gettempdir()

				if logging:
					logger.debug('FFmpeg Windows Download Path: {}'.format(ffmpeg_download_path))

				#download Binaries
				_path = download_ffmpeg_binaries(path = ffmpeg_download_path, os_windows = is_windows)
				#assign to local variable
				final_path += _path

			except Exception as e:
				#log if any error occurred
				if logging:
					logger.exception(str(e))
					logger.debug('Error in downloading FFmpeg binaries, Check your network and Try again!')
				return False

		if os.path.isfile(final_path):
			#check if valid FFmpeg file exist
			pass
		elif os.path.isfile(os.path.join(final_path, 'ffmpeg.exe')):
			#check if FFmpeg directory exists, if does, then check for valid file 
			final_path = os.path.join(final_path, 'ffmpeg.exe')
		else:
			#else return False
			if logging: logger.debug('No valid FFmpeg executables found at Custom FFmpeg path!')
			return False
	else:
		#otherwise perform test for Unix
		if custom_ffmpeg:
			#if custom FFmpeg path is given assign to local variable
			if os.path.isfile(custom_ffmpeg):
				#check if valid FFmpeg file exist
				final_path += custom_ffmpeg
			elif os.path.isfile(os.path.join(custom_ffmpeg, 'ffmpeg')):
				#check if FFmpeg directory exists, if does, then check for valid file 
				final_path = os.path.join(custom_ffmpeg, 'ffmpeg')
			else:
				#else return False
				if logging: logger.debug('No valid FFmpeg executables found at Custom FFmpeg path!')
				return False
		else:
			#otherwise assign ffmpeg binaries from system
			final_path += "ffmpeg"

	if logging:
		logger.debug('Final FFmpeg Path: {}'.format(final_path))

	# Final Auto-Validation for FFmeg Binaries. returns final path if test is passed
	if validate_ffmpeg(final_path, logging = logging):
		return final_path
	else:
		return False



def download_ffmpeg_binaries(path, os_windows = False):
	"""
	Download and Extract FFmpeg Static Binaries for windows(if not available)
	"""
	import platform
	final_path = ''
	if os_windows:
		windows_bit = 'win64' if platform.machine().endswith('64') else 'win32' #checks current Windows Bit Mode
		#inialize varibles
		file_url = 'https://ffmpeg.zeranoe.com/builds/{}/static/ffmpeg-latest-{}-static.zip'.format(windows_bit, windows_bit)
		file_name = os.path.join(os.path.abspath(path),'ffmpeg-latest-{}-static.zip'.format(windows_bit))
		file_path = os.path.join(os.path.abspath(path), 'ffmpeg-latest-{}-static/bin/ffmpeg.exe'.format(windows_bit))
		base_path, _ = os.path.split(file_name) #extract file base path
		#check if file already exists
		if os.path.isfile(file_path):
			final_path += file_path #skip download if does 
		else:
			#import libs
			import requests
			import zipfile
			#check if given path has write access
			assert os.access(path, os.W_OK), "[Helper:ERROR] :: Permission Denied, Cannot write binaries to directory = " + path
			#remove leftovers if exists
			if os.path.isfile(file_name): os.remove(file_name)
			#download and write file to the given path
			with open(file_name, "wb") as f:
				logger.debug("No Custom FFmpeg path provided. Auto-Installing FFmpeg static binaries now. Please wait...")
				try:
					response  = requests.get(file_url, stream=True, timeout=2)
					response.raise_for_status()
				except Exception as e:
					logger.exception(str(e))
					logger.warning("Downloading Failed. Trying GitHub mirror now!")
					file_url = 'https://raw.githubusercontent.com/abhiTronix/ffmpeg-static-builds/master/windows/ffmpeg-latest-{}-static.zip'.format(windows_bit, windows_bit)
					response  = requests.get(file_url, stream=True, timeout=2)
					response.raise_for_status()
				total_length = response.headers.get('content-length')
				if total_length is None: # no content length header
					f.write(response.content)
				else:
					dl = 0
					total_length = int(total_length)
					for data in response.iter_content(chunk_size=4096):
						dl += len(data)
						f.write(data)
						done = int(50 * dl / total_length)
						sys.stdout.write("\r[{}{}]{}{}".format('=' * done, ' ' * (50-done), done * 2, '%') )    
						sys.stdout.flush()
			logger.debug("Extracting executables.")
			with zipfile.ZipFile(file_name, "r") as zip_ref:
				zip_ref.extractall(base_path)
			#perform cleaning
			os.remove(file_name)
			logger.debug("FFmpeg binaries for Windows configured successfully!")
			final_path += file_path
	#return final path
	return final_path



def validate_ffmpeg(path, logging = False):
	"""
	Validate FFmeg Binaries. returns True if tests passed
	"""
	try:
		#get the FFmpeg version
		version = check_output([path, "-version"])
		firstline = version.split(b'\n')[0]
		version = firstline.split(b' ')[2].strip()
		if logging:
			#log if test are passed 
			logger.debug('FFmpeg validity Test Passed!')
			logger.debug('Found valid FFmpeg Version: `{}` installed on this system'.format(version))
	except Exception as e:
		#log if test are failed
		if logging:
			logger.exception(str(e))
			logger.debug('FFmpeg validity Test Failed!')
		return False
	return True



def check_output(*args, **kwargs):
	"""
	return output from the sub-process
	"""
	#silent subprocess execution
	closeNULL = 0
	import subprocess as sp
	try:
		from subprocess import DEVNULL
		closeNULL = 0
	except ImportError:
		DEVNULL = open(os.devnull, 'wb')
		closeNULL = 1
	#execute command in subprocess
	process = sp.Popen(stdout=sp.PIPE, stderr=DEVNULL, *args, **kwargs)
	output, unused_err = process.communicate()
	retcode = process.poll()
	#close the process
	if closeNULL:
		DEVNULL.close()
	#if error occurred raise error
	if retcode:
		cmd = kwargs.get("args")
		if cmd is None: cmd = args[0]
		error = sp.CalledProcessError(retcode, cmd)
		error.output = output
		raise error
	return output



def generate_auth_certificates(path, overwrite = False):

	""" 
	auto-Generates and auto-validates CURVE ZMQ keys/certificates for Netgear 
	"""

	#import necessary libs
	import shutil, errno
	import zmq.auth

	#check if path corresponds to vidgear only
	if (os.path.basename(path) != ".vidgear"): path = os.path.join(path,".vidgear")

	#generate keys dir
	keys_dir = os.path.join(path, 'keys')
	try:
		os.makedirs(keys_dir)
	except OSError as e:
		if e.errno != errno.EEXIST: raise

	#generate separate public and private key dirs
	public_keys_dir = os.path.join(keys_dir, 'public_keys')
	secret_keys_dir = os.path.join(keys_dir, 'private_keys')

	#check if overwriting is allowed
	if overwrite:
		#delete previous certificates
		for d in [public_keys_dir, secret_keys_dir]:
			if os.path.exists(d): shutil.rmtree(d)
			os.mkdir(d)

		# generate new keys
		server_public_file, server_secret_file = zmq.auth.create_certificates(keys_dir, "server")
		client_public_file, client_secret_file = zmq.auth.create_certificates(keys_dir, "client")

		# move keys to their appropriate directory respectively
		for key_file in os.listdir(keys_dir):
			if key_file.endswith(".key"):
				shutil.move(os.path.join(keys_dir, key_file), public_keys_dir)
			elif key_file.endswith(".key_secret"):
				shutil.move(os.path.join(keys_dir, key_file), secret_keys_dir)
			else:
				# clean redundant keys if present
				redundant_key = os.path.join(keys_dir,key_file)
				if os.path.isfile(redundant_key):
					os.remove(redundant_key)
	else:
		# otherwise validate available keys
		status_public_keys = validate_auth_keys(public_keys_dir, '.key')
		status_private_keys = validate_auth_keys(secret_keys_dir, '.key_secret')

		# check if all valid keys are found
		if status_private_keys and status_public_keys: return (keys_dir, secret_keys_dir, public_keys_dir)

		# check if valid public keys are found
		if not(status_public_keys):
			try:
				os.makedirs(public_keys_dir)
			except OSError as e:
				if e.errno != errno.EEXIST: raise

		# check if valid private keys are found
		if not(status_private_keys): 
			try:
				os.makedirs(secret_keys_dir)
			except OSError as e:
				if e.errno != errno.EEXIST: raise

		# generate new keys
		server_public_file, server_secret_file = zmq.auth.create_certificates(keys_dir, "server")
		client_public_file, client_secret_file = zmq.auth.create_certificates(keys_dir, "client")

		# move keys to their appropriate directory respectively
		for key_file in os.listdir(keys_dir):
			if key_file.endswith(".key") and not(status_public_keys):
				shutil.move(os.path.join(keys_dir, key_file), os.path.join(public_keys_dir, '.'))
			elif key_file.endswith(".key_secret") and not(status_private_keys):
				shutil.move(os.path.join(keys_dir, key_file), os.path.join(secret_keys_dir, '.'))
			else:
				# clean redundant keys if present
				redundant_key = os.path.join(keys_dir,key_file)
				if os.path.isfile(redundant_key): os.remove(redundant_key)

	# validate newly generated keys 
	status_public_keys = validate_auth_keys(public_keys_dir, '.key')
	status_private_keys = validate_auth_keys(secret_keys_dir, '.key_secret')

	# raise error is validation test fails
	if not(status_private_keys) or not(status_public_keys): raise RuntimeError('[Helper:ERROR] :: Unable to generate valid ZMQ authentication certificates at `{}`!'.format(keys_dir))

	# finally return valid key paths
	return (keys_dir, secret_keys_dir, public_keys_dir)



def validate_auth_keys(path, extension):

	"""
	validates and maintains ZMQ Auth Keys/Certificates
	"""
	#check for valid path
	if not(os.path.exists(path)): return False
	
	#check if directory empty
	if not(os.listdir(path)): return False

	keys_buffer = [] #stores auth-keys

	# loop over auth-keys
	for key_file in os.listdir(path):
		key = os.path.splitext(key_file)
		#check if valid key is generated
		if key and (key[0] in ['server','client']) and (key[1] == extension): keys_buffer.append(key_file) #store it

	#remove invalid keys if found
	if(len(keys_buffer) == 1): os.remove(os.path.join(path,keys_buffer[0])) 

	#return results
	return True if(len(keys_buffer) == 2) else False

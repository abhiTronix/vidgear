"""
============================================
vidgear library code is placed under the MIT license
Copyright (c) 2019 Abhishek Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
"""

# Contains all the support functions/modules required by Vidgear 


# import the neccesary packages
import os, sys
import cv2
	

def check_python_version():
	"""
	returns current python version's - first bit 
	"""
	return sys.version_info[0]


def check_CV_version(self):
	"""
	returns current OpenCV binary version's - first bit 
	"""
	if parse_version(cv2.__version__) >= parse_version('4'):
		return 4
	else:
		return 3


def capPropId(property):
	"""
	Retrieves the Property's Integer(Actual) value. 
	"""
	return getattr(cv2, property)


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
				if ffmpeg_download_path:
					#checks if FFmpeg download path specified
					pass
				else:
					#otherwise save to Temp Directory
					import tempfile
					ffmpeg_download_path = tempfile.gettempdir()

				if logging:
					print('FFmpeg Windows Download Path: {}'.format(ffmpeg_download_path))

				#download Binaries
				_path = download_ffmpeg_binaries(path = ffmpeg_download_path, os_windows = is_windows)
				#assign to local variable
				final_path += _path

			except Exception as e:
				#log if any error occured
				if logging:
					print(e)
					print('Error downloading FFmpeg binaries, Check your network and Try again!')
				return False
		if os.path.isfile(final_path):
			#check if valid FFmpeg file exist
			pass
		elif os.path.isfile(os.path.join(final_path, 'ffmpeg.exe')):
			#check if FFmpeg directory exists, if does, then check for valid file 
			final_path = os.path.join(final_path, 'ffmpeg.exe')
		else:
			#else return False
			if logging:
				print('No valid FFmpeg executables found at Custom FFmpeg path!')
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
				if logging:
					print('No valid FFmpeg executables found at Custom FFmpeg path!')
				return False
		else:
			#otherwise assign ffmpeg binaries from system
			final_path += "ffmpeg"

	if logging:
		print('Final FFmpeg Path: {}'.format(final_path))

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
			import requests
			import zipfile
			#check if given pth has write access
			assert os.access(path, os.W_OK), "Permission Denied: Cannot write ffmpeg binaries to directory = " + path
			#remove leftovers
			if os.path.isfile(file_name):
				os.remove(file_name)
			#download and write file to the given path
			with open(file_name, "wb") as f:
				print("No Custom FFmpeg path provided, Auto-Downloading binaries for Windows. Please wait...")
				response  = requests.get(file_url, stream=True)
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
			print("\nExtracting executables, Please Wait...")
			with zipfile.ZipFile(file_name, "r") as zip_ref:
				zip_ref.extractall(base_path)
			#perform cleaning
			os.remove(file_name)
			print("FFmpeg binaries for Windows Configured Successfully!")
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
			print('FFmpeg validity Test Passed!')
			print('Found valid FFmpeg Version: `{}` installed on this system'.format(version))
	except Exception as e:
		#log if test are failed
		if logging:
			print(e)
			print('FFmpeg validity Test Failed!')
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
	#if error occured raise error
	if retcode:
		cmd = kwargs.get("args")
		if cmd is None:
			cmd = args[0]
		error = sp.CalledProcessError(retcode, cmd)
		error.output = output
		raise error
	return output
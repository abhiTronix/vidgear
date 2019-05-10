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

# import the necessary packages/libraries
from pkg_resources import parse_version
import os, sys, time
import subprocess as sp

from .helper import get_valid_ffmpeg_path
from .helper import capPropId
from .helper import dict2Args

try:
	# import OpenCV Binaries
	import cv2
	# check whether OpenCV Binaries are 3.x+
	if parse_version(cv2.__version__) >= parse_version('3'):
		pass
	else:
		raise ImportError('OpenCV library version >= 3.0 is only supported by this library')
except ImportError as error:
	raise ImportError('Failed to detect OpenCV executables, install it with `pip install opencv-contrib-python` command.')


class WriteGear:
	"""
	WriteGear class contains advanced tools that enables High-Speed Lossless Video Encoding with Compression capabilities within Vidgear. 
	This class basically pipelines, processes, encodes and finally writes video frames into a valid Video File.

	In Compression Mode, this class utilizes FFmpeg(a powerful tool that can do almost anything you can imagine with multimedia files) 
	 powerful encoders to encode and reduce the output to a smaller size, without sacrificing the video quality. This class give us silent 
	and flexible control to use all the parameters available within FFmpeg to manipulate the properties (quality, codecs, 
	format and size etc.) of output video and it robustly handles errors/warnings on the way.

	This also supports Simple OpenCV's inbuilt `VideoWriter class` in Non-Compression Mode. Also the parameters/properties of 
	OpenCV VideoWriter can also be manipulated as similar to Compression Mode, but this mode lacks ability to compress video while encoding. 

	```Warning: In case, This class fails to detect valid FFmpeg executables on your system, It can automatically fallbacks to Non-Compression Mode.```

	:param output_filename (string): sets the output Video file path/filename. Valid Inputs are:

		- path: Valid path of the directory to save the output video file. Then, the class will automatically assign filename(with default extension `.mp4`).  

		- filename(with path): Valid filename(with valid extension) of the output video file. 

	```Note: Make sure to provide valid filename with path and file-extension(based on encoder(in case of FFmpeg)/fourcc(in case of OpenCV) being currently used)```

	:param compression_mode (boolean): enables/disables Video Compression Mode, i.e If enabled(means Compression Mode) WriteGear class will utilize 
						/ the FFmpeg installed/given executables to encode output video and if disabled(means Non-Compression Mode) 
						/ OpenCV's VideoWriter Class will be used. Its default value is True.

	:param custom_ffmpeg (string): (if specified) sets the custom path/directory where the `Custom FFmpeg executables` are located, works only in Compression Mode.
						/ It is compulsory to provide custom binaries on a Windows Machine, otherwise this class with automatically download and extract suitable Static FFmpeg 
						/ binaries to Temporary directory.Read VidGear Docs for more info.

	:param logging (boolean): set this flag to enable/disable error logging essential for debugging. Its default value is False.

	:param **output_params (dict): sets all properties supported by FFmpeg(in Compression Mode)/ OpenCV's VideoWriter(in Non-Compression Mode) properties 
						/ to the input video stream in CamGear Class. All the valid User-defined FFmeg/OpenCV's VideoWriter output video file parameters/properties 
						/ manipulation is currently supported. These attribute provides the flexibility to manipulate output video  directly. 
						/ In addition other special attributes to manipulate WriteGear inbuilt properties are also supported. Read VidGear Docs for more info.

	"""

	def __init__(self, output_filename = '', compression_mode = True , custom_ffmpeg = '', logging = False, **output_params):

		# assign parameter values to class variables
		self.compression = compression_mode
		self.os_windows  = True if os.name == 'nt' else False #checks if machine in-use is running windows or not
		self.logging = logging

		# initialize various important class variables
		self.output_parameters = {}
		self.inputheight = None
		self.inputwidth = None
		self.inputchannels = None
		self.inputframerate = 0
		self.output_dimensions = None
		self.process = None #handle process to be frames written
		self.DEVNULL = None #handles silent execution of FFmpeg (if logging is disabled)
		self.cmd = ''     #handle FFmpeg Pipe command
		self.ffmpeg = ''  #handle valid FFmpeg binaries location
		self.initiate = True #initiate one time process for valid process initialization


		# handles output file name (if not given)
		if not output_filename:
			raise ValueError('Kindly provide a valid `output_filename` value, Refer VidGear Docs for more information!')
		elif output_filename and os.path.isdir(output_filename): # check if directory path is given instead
			output_filename = os.path.join(output_filename, 'VidGear-{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S"))) # auto-assign valid name and adds it to path
		else:
			pass

		# some definitions and assigning output file absolute path to class variable 
		_filename = os.path.abspath(output_filename)
		self.out_file = _filename
		basepath, _ = os.path.split(_filename) #extract file base path for debugging ahead


		if output_params:
			#handle user defined output dimensions(must be a tuple or list)
			if output_params and "-output_dimensions" in output_params:
				self.output_dimensions = output_params["-output_dimensions"] #assign special parameter to global variable
				del output_params["-output_dimensions"] #clean
			#cleans and reformat output parameters
			try:
				self.output_parameters = {str(k).strip().lower(): str(v).strip().lower() for k,v in output_params.items()}
			except Exception as e:
				if self.logging:
					print(e)
				raise ValueError('Wrong output_params parameters passed to WriteGear class!')

		#handles FFmpeg binaries validity tests 
		if self.compression:

			if self.logging:
				print('Compression Mode is enabled therefore checking for valid FFmpeg executables!')
				print(self.output_parameters)

			# handles where to save the downloaded FFmpeg Static Binaries on Windows(if specified)
			ffmpeg_download_path_ = ''
			if self.output_parameters and "-ffmpeg_download_path" in self.output_parameters:
				ffmpeg_download_path_ += self.output_parameters["-ffmpeg_download_path"]
				del self.output_parameters["-ffmpeg_download_path"] #clean

			#handle input framerate if specified
			if self.output_parameters and "-input_framerate" in self.output_parameters:
				self.inputframerate += float(self.output_parameters["-input_framerate"])
				del self.output_parameters["-input_framerate"] #clean

			#validate the FFmpeg path/binaries and returns valid FFmpeg file executable location(also downloads static binaries on windows) 
			actual_command = get_valid_ffmpeg_path(custom_ffmpeg, self.os_windows, ffmpeg_download_path = ffmpeg_download_path_, logging = self.logging)

			#check if valid path returned
			if actual_command:
				self.ffmpeg += actual_command #assign it to class variable
				if self.logging:
					print('Found valid FFmpeg executables: `{}`'.format(self.ffmpeg))
			else:
				#otherwise disable Compression Mode
				if self.logging and not self.os_windows:
					print('Kindly install working FFmpeg or provide a valid custom FFmpeg Path')
				print('Caution: Disabling Video Compression Mode since no valid FFmpeg executables found on this machine!')
				self.compression = False # compression mode disabled

		#validate this class has the access rights to specified directory or not
		assert os.access(basepath, os.W_OK), "Permission Denied: Cannot write to directory = " + basepath

		#display confirmation if logging is enabled/disabled
		if self.compression and self.ffmpeg:
			self.DEVNULL = open(os.devnull, 'wb') 
			if self.logging:
				print('Compression Mode is configured properly!')
		else:
			if self.logging:
				print('Compression Mode is disabled, Activating OpenCV In-built Writer!')




	def write(self, frame, rgb_mode = False):

		"""  
		pipelines ndarray frames to valid Video File. Utilizes FFmpeg in Compression Mode and OpenCV VideoWriter Class in Non-Compression Mode

		:param frame (ndarray): valid frame
		:param rgb_mode (boolean): set this flag to enable rgb_mode, i.e. specifies that incoming frames are of RGB format(instead of default BGR). Its default value is False.

		"""
		if frame is None: #NoneType frames will be skipped 
			return

		#get height, width and number of channels of current frame
		height, width = frame.shape[:2]
		channels = frame.shape[-1] if frame.ndim == 3 else 1

		# assign values to class variables on first run 
		if self.initiate:
			self.inputheight = height
			self.inputwidth = width
			self.inputchannels = channels
			if self.logging:
				print('InputFrame => Height:{} Width:{} Channels:{}'.format(self.inputheight, self.inputwidth, self.inputchannels))

		#validate size of frame
		if height != self.inputheight or width != self.inputwidth:
			raise ValueError('All frames in a video should have same size')
		#validate number of channels
		if channels != self.inputchannels:
			raise ValueError('All frames in a video should have same number of channels')

		if self.compression:
			# checks if compression mode is enabled

			#initiate FFmpeg process on first run
			if self.initiate:
				#start pre-processing and initiate process 
				self.Preprocess(channels, rgb = rgb_mode)
				# Check status of the process
				assert self.process is not None

			#write the frame
			try:
				self.process.stdin.write(frame.tostring())
			except (OSError, IOError):
				# log something is wrong!
				print ('BrokenPipeError caught: Wrong Values passed to FFmpeg Pipe, Kindly Refer Docs!')
				sys.stderr.close()
				raise ValueError #for testing purpose only
		else:
			# otherwise initiate OpenCV's VideoWriter Class
			if self.initiate:
				#start VideoWriter Class process
				self.startCV_Process()
				# Check status of the process
				assert self.process is not None
				if self.logging:
					# log OpenCV warning
					print('Warning: RGBA and 16-bit grayscale video frames are not supported by OpenCV yet, switch to `compression_mode` to use them!')
			#write the frame
			self.process.write(frame)



	def Preprocess(self, channels, rgb = False):
		"""
		pre-processing FFmpeg parameters
		
		:param channels (int): Number of channels
		:param rgb_mode (boolean): set this flag to enable rgb_mode, Its default value is False.
		"""

		#turn off initiate flag
		self.initiate = False
		#initialize input parameters
		input_parameters = {}

		#handle dimensions
		dimensions = ''
		if self.output_dimensions is None: #check if dimensions are given
			dimensions += '{}x{}'.format(self.inputwidth, self.inputheight) #auto derive from frame
		else:
			dimensions += '{}x{}'.format(self.output_dimensions[0],self.output_dimensions[1]) #apply if defined
		input_parameters["-s"] = str(dimensions)

		#handles pix_fmt based on channels(HACK)
		if channels == 1:
			input_parameters["-pix_fmt"] = "gray"
		elif channels == 2:
			input_parameters["-pix_fmt"] = "ya8"
		elif channels == 3:
			input_parameters["-pix_fmt"] = "rgb24" if rgb else "bgr24"
		elif channels == 4:
			input_parameters["-pix_fmt"] = "rgba" if rgb else "bgra"
		else:
			raise ValueError("Handling Frames with (1 > Channels > 4) is not implemented!")

		if self.inputframerate > 5:
			#set input framerate - minimum threshold is 5.0
			if self.logging:
				print("Setting Input FrameRate = {}".format(self.inputframerate))
			input_parameters["-framerate"] = str(self.inputframerate)

		#initiate FFmpeg process
		self.startFFmpeg_Process(input_params = input_parameters, output_params = self.output_parameters)



	def startFFmpeg_Process(self, input_params, output_params):

		"""
		Start FFmpeg process

		:param input_params (dict): Input parameters
		:param output_params (dict): Output parameters
		"""

		#convert input parameters to list
		input_parameters = dict2Args(input_params)

		#pre-assign default encoder parameters (if not assigned by user).
		if "-vcodec" not in output_params:
			output_params["-vcodec"] = "libx264"
		if output_params["-vcodec"] in ["libx264", "libx265"]:
			if "-crf" in output_params: 
				pass
			else:
				output_params["-crf"] = "18"
			if "-preset" in output_params: 
				pass
			else:
				output_params["-preset"] = "fast"
		#convert output parameters to list
		output_parameters = dict2Args(output_params)
		#format command
		cmd = [self.ffmpeg , "-y"] + ["-f", "rawvideo", "-vcodec", "rawvideo"] + input_parameters + ["-i", "-"] + output_parameters + [self.out_file]
		#assign value to class variable
		self.cmd += " ".join(cmd)
		# Launch the FFmpeg process
		if self.logging:
			print(self.cmd)
			# In debugging mode
			self.process = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
		else:
			# In silent mode
			self.process = sp.Popen(cmd, stdin=sp.PIPE, stdout=self.DEVNULL, stderr=sp.STDOUT)


	def startCV_Process(self):
		"""
		Start OpenCV VideoWriter Class process
		"""

		#turn off initiate flag
		self.initiate = False

		#initialize essential parameter variables
		FPS = 0
		BACKEND = ''
		FOURCC = 0
		COLOR = True

		#pre-assign default encoder parameters (if not assigned by user).
		if "-fourcc" not in self.output_parameters:
			FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
		if "-fps" not in self.output_parameters:
			FPS = 25

		#auto assign dimensions	
		HEIGHT = self.inputheight
		WIDTH = self.inputwidth

		#assign parameter dict values to variables
		try:
			for key, value in self.output_parameters.items():
				if key == '-fourcc':
					FOURCC = cv2.VideoWriter_fourcc(*(value.upper()))
				elif key == '-fps':
					FPS = float(value)
				elif key =='-backend' and value.upper() in ['CAP_FFMPEG','CAP_GSTREAMER']:
					BACKEND = capPropId(value.upper())
				elif key == '-color':
					COLOR = bool(int(value))
				else:
					pass

		except Exception as e:
			# log if something is wrong
			if self.logging:
				print(e)
			raise ValueError('Wrong Values passed to OpenCV Writer, Kindly Refer Docs!')

		if self.logging:
			#log values for debugging
			print('FILE_PATH: {}, FOURCC = {}, FPS = {}, WIDTH = {}, HEIGHT = {}, BACKEND = {}'.format(self.out_file,FOURCC, FPS, WIDTH, HEIGHT, BACKEND))

		#start different process for with/without Backend.
		if BACKEND: 
			self.process = cv2.VideoWriter(self.out_file, apiPreference = BACKEND, fourcc = FOURCC, fps = FPS, frameSize = (WIDTH, HEIGHT), isColor = COLOR)
		else:
			self.process = cv2.VideoWriter(self.out_file, fourcc = FOURCC, fps = FPS, frameSize = (WIDTH, HEIGHT), isColor = COLOR)


	def close(self):
		"""
		Terminates the Write process
		"""
		if self.compression:
			#if Compression Mode is enabled
			if self.process is None:  
				return  #no process was initiated at first place
			if self.process.poll() is not None: 
				return  # process was already dead
			if self.process.stdin:
				self.process.stdin.close() #close `stdin` output
			self.process.wait() #wait if still process is still processing some information
			self.process = None 
			self.DEVNULL.close() #close it
		else:
			#if Compression Mode is disabled
			if self.process is None: 
				return  #no process was initiated at first place
			self.process.release() #close it


	






		

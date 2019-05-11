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

# import the necessary packages
from threading import Thread
from pkg_resources import parse_version
from .helper import capPropId
from .helper import check_CV_version
import re

#Note: Remember, Not all parameters are supported by all cameras which is 
#one of the most troublesome part of the OpenCV library. Each camera type, 
#from android cameras to USB cameras to professional
#ones offer a different interface to modify its parameters. 
#Therefore there are many branches in OpenCV code to support 
#as many of them, but of course not all possible devices 
#are covered and thereby works.

#To check parameter values supported by your webcam, you can hook your camera
#to your Linux machine and use command `v4l2-ctl -d 0 --list-formats-ext` (where 0 is index of given camera)
#to list the supported video parameters and their values.

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


def youtube_url_validation(url):
	"""
	convert youtube video url and checks its validity
	"""
	youtube_regex = (
		r'(https?://)?(www\.)?'
		'(youtube|youtu|youtube-nocookie)\.(com|be)/'
		'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
	youtube_regex_match = re.match(youtube_regex, url)
	if youtube_regex_match:
		return youtube_regex_match.group(6)
	return youtube_regex_match


class CamGear:
	"""
	This class targets any common IP or USB Cameras(including Raspberry Pi Compatible), 
	Various Video Files Formats and Network Video Streams(Including Gstreamer Raw Video Capture Pipeline) 
	for obtaining high-speed real-time frames by utilizing OpenCV and multi-threading. Now, it also supports Youtube Streaming.

	:param source : take the source value. Its default value is 0. Valid Inputs are:

		- Index(integer): Valid index of the video device.

		- YouTube Url(string): Valid Youtube URL as input.

		- Network_Stream_Address(string): Incoming Stream Valid Network address. 

		- GStreamer (string) videostream Support


	:param (boolean) y_tube: enables YouTube Mode, i.e If enabled class will interpret the given source string as YouTube URL. Its default value is False.

	:param (int) backend: set the backend of the video stream (if specified). Its default value is 0.

	:param (string) colorspace: set the colorspace of the video stream. Its default value is None.

	:param (dict) **options: sets all properties supported by OpenCV's VideoCapture Class properties to the input video stream in CamGear Class. 
					  / These attribute provides the flexibility to manipulate input webcam video stream directly. 
					  / Parameters can be passed using this **option, allows you to pass keyworded variable length of arguments to CamGear Class.

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	:param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
						/ This delay is essentially required for camera to warm-up. 
						/Its default value is 0.
	"""
	def __init__(self, source = 0, y_tube = False, backend = 0, colorspace = None, logging = False, time_delay = 0, **options):


		# check if Youtube Mode is ON (True)
		if y_tube:
			try:
				#import pafy and parse youtube stream url
				import pafy
				# validate
				url = youtube_url_validation(source)
				if url:
					source_object = pafy.new(url)
					_source = source_object.getbestvideo("any", ftypestrict=False)
					if logging:
						print('Title: {}'.format(source_object.title))
						print('Extension: {}'.format(_source.extension))
					source = _source.url
			except Exception as e:
				if logging:
					print(e)
				raise ValueError('YouTube Mode is enabled and the input YouTube Url is invalid!')

		# stream variable initialization
		self.stream = None

		if backend and isinstance(backend, int):
			# add backend if scpecified and initialize the camera stream
			if check_CV_version() == 3:
				# Different OpenCV 3.4.x statement
				self.stream = cv2.VideoCapture(source + backend)
			else:
				# Two parameters are available since OpenCV 4+ (master branch)
				self.stream = cv2.VideoCapture(source, backend)
		else:
			# initialize the camera stream
			self.stream = cv2.VideoCapture(source)


		#initialisation colorspace variable
		self.color_space = None

		#reformat dict
		options = {k.strip(): v for k,v in options.items()}


		try: 
			# try to apply attributes to source if specified
			for key, value in options.items():
				self.stream.set(capPropId(key),value)

			# seperately handle colorspace value to int conversion
			if not(colorspace is None):
				self.color_space = capPropId(colorspace.strip())

		except Exception as e:
			# Catch if any error occurred
			if logging:
				print(e)

		#initialize framerate variable
		self.framerate = 0

		#save framerate
		try:
			_fps = self.stream.get(cv2.CAP_PROP_FPS)
			if _fps>1:
				self.framerate = _fps
		except Exception as e:
			if logging:
				print(e)
			self.framerate = 0

		(grabbed, self.frame) = self.stream.read()

		# applying time delay to warm-up webcam only if specified
		if time_delay:
			import time
			time.sleep(time_delay)

		# enable logging if specified
		self.logging = logging

		# thread initialization
		self.thread=None

		# initialize termination flag
		self.terminate = False

	def start(self):
		"""
		start the thread to read frames from the video stream
		"""
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()
		return self

	def update(self):
		"""
		Update frames from stream
		"""
		# keep looping infinitely until the thread is terminated
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.terminate:
				break

			# otherwise, read the next frame from the stream
			(grabbed, frame) = self.stream.read()

			#check for valid frames
			if not grabbed:
				#no frames received, then safely exit
				self.terminate = True

			if not(self.color_space is None):
				# apply colorspace to frames
				color_frame = None
				try:
					if isinstance(self.color_space, int):
						color_frame = cv2.cvtColor(frame, self.color_space)
					else:
						self.color_space = None
						if self.logging:
							print('Colorspace value {} is not a valid Colorspace!'.format(self.color_space))
				except Exception as e:
					# Catch if any error occurred
					self.color_space = None
					if self.logging:
						print(e)
						print('Input Colorspace is not a valid Colorspace!')

				if not(color_frame is None):
					self.frame = color_frame
				else:
					self.frame = frame
			else:
				self.frame = frame
				
		#release resources
		self.stream.release()

	def read(self):
		"""
		return the frame
		"""
		return self.frame

	def stop(self):
		"""
		Terminates the Read process
		"""
		# indicate that the thread should be terminate
		self.terminate = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		if self.thread is not None: 
			self.thread.join()
			#properly handle thread exit

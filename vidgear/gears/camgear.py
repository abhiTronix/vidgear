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

# import the necessary packages
from threading import Thread
from pkg_resources import parse_version
from .helper import capPropId
from .helper import check_CV_version
import re, time
import logging as log



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
	if parse_version(cv2.__version__) < parse_version('3'):
		raise ImportError('[CamGear:ERROR] :: OpenCV API version >= 3.0 is only supported by this library.')
except ImportError as error:
	raise ImportError('[CamGear:ERROR] :: Failed to detect correct OpenCV executables, install it with `pip3 install opencv-python` command.')



def youtube_url_validation(url):
	"""
	convert Youtube video URLs to a valid address
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
	CamGear API supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia 
	video file format (upto 4k tested), network stream URL such as http(s), rtp, rstp, mms, etc. In addition to this, it also supports 
	live Gstreamer's RAW pipelines and YouTube video/livestreams URLs. CamGear provides a flexible, high-level multi-threaded wrapper 
	around OpenCV's VideoCapture API with access almost all of its available parameters and also employs pafy's APIs for YouTube streaming. 
	Furthermore, CamGear relies exclusively on Threaded Queue mode for ultra-fast, error-free and synchronized frame handling.  

	Threaded Queue Mode => Sequentially adds and releases frames to/from deque and handles overflow of this queue. It utilizes 
	Deques that support thread-safe, memory efficient appends and pops from either side of the deque with approximately the 
	same O(1) performance in either direction.  

	:param source : take the source value. Its default value is 0. Valid Inputs are:

		- Index(integer): Valid index of the video device.

		- YouTube Url(string): Valid Youtube URL as input.

		- Network_Stream_Address(string): Incoming Stream Valid Network address. 

		- GStreamer (string) pipeline

	:param (boolean) y_tube: enables YouTube Mode, i.e If enabled class will interpret the given source string as YouTube URL. Its default value is False.

	:param (int) backend: set the backend of the video stream (if specified). Its default value is 0.

	:param (string) colorspace: set the colorspace of the video stream. Its default value is None.

	:param (dict) **options: provides the ability to tweak properties supported by OpenCV's VideoCapture 
							API properties for any given input video stream directly. All the supported 
							parameters can be passed to CamGear API using this dict as follows:

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	:param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
						/ This delay is essentially required for camera to warm-up. 
						/Its default value is 0.
	"""

	def __init__(self, source = 0, y_tube = False, backend = 0, colorspace = None, logging = False, time_delay = 0, **options):

		#intialize threaded queue mode
		self.__threaded_queue_mode = True

		# enable logging if specified
		self.__logging = False
		self.__logger = log.getLogger('CamGear')
		if logging: self.__logging = logging

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
					if _source is None: _source = source_object.getbest("any", ftypestrict=False)
					if self.__logging: self.__logger.debug('YouTube source ID: `{}`, Title: `{}` & Video_Extension: `{}`'.format(url, source_object.title, _source.extension))
					source = _source.url
				else: raise RuntimeError('URL cannot be processed!')
			except Exception as e:
				if self.__logging: self.__logger.exception(str(e))
				raise ValueError('[CamGear:ERROR] :: YouTube Mode is enabled and the input YouTube URL is invalid!')

		# youtube mode variable initialization
		self.__youtube_mode = y_tube

		#User-Defined Threaded Queue Mode
		if options:
			if "THREADED_QUEUE_MODE" in options:
				if isinstance(options["THREADED_QUEUE_MODE"],bool):
					self.__threaded_queue_mode = options["THREADED_QUEUE_MODE"] #assigsn special parameter to global variable
				del options["THREADED_QUEUE_MODE"] #clean
				#reformat option dict

		self.__queue = None
		#intialize deque for video files only 
		if self.__threaded_queue_mode and isinstance(source,str):
			#import deque
			from collections import deque
			#define deque and assign it to global var
			self.__queue = deque(maxlen=96) #max len 96 to check overflow
			#log it
			if self.__logging: self.__logger.debug('Enabling Threaded Queue Mode for the current video source!') 
		else:
			#otherwise disable it
			self.__threaded_queue_mode = False
			#log it
			if self.__logging: self.__logger.debug('Threaded Queue Mode is disabled for the current video source!') 

		# stream variable initialization
		self.stream = None

		if backend and isinstance(backend, int):
			# add backend if specified and initialize the camera stream
			if check_CV_version() == 3:
				# Different OpenCV 3.4.x statement
				self.stream = cv2.VideoCapture(source + backend)
			else:
				# Two parameters are available since OpenCV 4+ (master branch)
				self.stream = cv2.VideoCapture(source, backend)
		else:
			# initialize the camera stream
			self.stream = cv2.VideoCapture(source)


		#initializing colorspace variable
		self.color_space = None

		try: 
			# try to apply attributes to source if specified
			#reformat dict
			options = {k.strip(): v for k,v in options.items()}
			for key, value in options.items():
				self.stream.set(capPropId(key),value)

			# separately handle colorspace value to int conversion
			if not(colorspace is None): 
				self.color_space = capPropId(colorspace.strip())
				if self.__logging: self.__logger.debug('Enabling `{}` colorspace for this video stream!'.format(colorspace.strip()))

		except Exception as e:
			# Catch if any error occurred
			if self.__logging: self.__logger.exception(str(e))

		#initialize and assign framerate variable
		self.framerate = 0.0
		try:
			_fps = self.stream.get(cv2.CAP_PROP_FPS)
			if _fps>1: self.framerate = _fps
		except Exception as e:
			if self.__logging: self.__logger.exception(str(e))
			self.framerate = 0.0

		# applying time delay to warm-up webcam only if specified
		if time_delay: time.sleep(time_delay)

		#frame variable initialization
		(grabbed, self.frame) = self.stream.read()

		#check if vaild stream
		if grabbed:
			#render colorspace if defined
			if not(self.color_space is None): self.frame = cv2.cvtColor(self.frame, self.color_space)

			if self.__threaded_queue_mode:
				#intitialize and append to queue
				self.__queue.append(self.frame)
		else:
			raise RuntimeError('[CamGear:ERROR] :: Source is invalid, CamGear failed to intitialize stream on this source!')

		# thread initialization
		self.__thread=None

		# initialize termination flag
		self.__terminate = False



	def start(self):
		"""
		start the thread to read frames from the video stream
		"""
		self.__thread = Thread(target=self.__update, name='CamGear', args=())
		self.__thread.daemon = True
		self.__thread.start()
		return self



	def __update(self):
		"""
		Update frames from stream
		"""

		# keep iterating infinitely until the thread is terminated or frames runs out
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.__terminate:
				break

			if self.__threaded_queue_mode:
				#check queue buffer for overflow
				if len(self.__queue) >= 96:
					#stop iterating if overflowing occurs
					time.sleep(0.000001)
					continue

			# otherwise, read the next frame from the stream
			(grabbed, frame) = self.stream.read()

			#check for valid frames
			if not grabbed:
				#no frames received, then safely exit
				if self.__threaded_queue_mode:
					if len(self.__queue) == 0: 
						break
					else:
						continue
				else:
					break

			if not(self.color_space is None):
				# apply colorspace to frames
				color_frame = None
				try:
					if isinstance(self.color_space, int):
						color_frame = cv2.cvtColor(frame, self.color_space)
					else:
						self.color_space = None
						if self.__logging: self.__logger.warning('Colorspace value: {}, is not a valid colorspace!'.format(self.color_space))
				except Exception as e:
					# Catch if any error occurred
					self.color_space = None
					if self.__logging:
						self.__logger.exception(str(e))
						self.__logger.warning('Input colorspace is not a valid colorspace!')
				if not(color_frame is None):
					self.frame = color_frame
				else:
					self.frame = frame
			else:
				self.frame = frame

			#append to queue
			if self.__threaded_queue_mode: self.__queue.append(self.frame)

		self.__threaded_queue_mode = False
		self.frame = None
		#release resources
		self.stream.release()



	def read(self):
		"""
		return the frame
		"""
		while self.__threaded_queue_mode:
			if len(self.__queue) > 0: 
				return self.__queue.popleft()
		return self.frame



	def stop(self):
		"""
		Terminates the Read process
		"""
		if self.__logging: self.__logger.debug('Terminating processes.')
		#terminate Threaded queue mode seperately
		if self.__threaded_queue_mode and not(self.__queue is None):
			if len(self.__queue)>0: self.__queue.clear()
			self.__threaded_queue_mode = False
			self.frame = None

		# indicate that the thread should be terminate
		self.__terminate = True

		# wait until stream resources are released (producer thread might be still grabbing frame)
		if self.__thread is not None:
			self.__thread.join()
			#properly handle thread exit
			if self.__youtube_mode:
				# kill thread-lock in youtube mode
				self.__thread = None
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
import numpy as np
import time
import logging as log


try:
	# import OpenCV Binaries
	import cv2
	# check whether OpenCV Binaries are 3.x+
	if parse_version(cv2.__version__) < parse_version('3'):
		raise ImportError('[ScreenGear:ERROR] :: OpenCV library version >= 3.0 is only supported by this library')

except ImportError as error:
	raise ImportError('[ScreenGear:ERROR] :: Failed to detect OpenCV executables, install it with `pip install opencv-python` command.')



class ScreenGear:

	"""
	With ScreenGear, we can easily define an area on the computer screen or an open window to record the live screen frames in 
	real-time at the expense of insignificant latency. To achieve this, ScreenGear provides a high-level multi-threaded wrapper 
	around mss python library API and also supports the flexible direct parameter manipulation. Furthermore, ScreenGear relies on 
	Threaded Queue mode for ultra-fast live frame handling and which is enabled by default.

	Threaded Queue Mode => Sequentially adds and releases frames to/from deque and handles overflow of this queue. It utilizes 
	Deques that support thread-safe, memory efficient appends and pops from either side of the deque with approximately the 
	same O(1) performance in either direction.  


	:param monitor(int): sets the Positions/Location of monitor where to grab frame from. More information can be found in the docs. 
						/ It default value is 1 (means current monitor will be used).

	:param **options(dict): can be used to pass parameters to ScreenGear Class. 
							/This attribute provides the flexibility to manipulate mss input parameters 
							/directly like the dimensions of the region of the given monitor from where the 
							frames to be grabbed. Checkout VidGear docs for usage details.

	:param (string) colorspace: set the colorspace of the video stream. Its default value is None.

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	"""
	
	def __init__(self, monitor = 1, colorspace = None, logging = False, **options):

		#intialize threaded queue mode by default
		self.threaded_queue_mode = True

		try:
			# import mss factory
			from mss import mss
			# import mss error handler
			from mss.exception import ScreenShotError
		except ImportError as error:
			# otherwise raise import error
			raise ImportError('[ScreenGear:ERROR] :: python-mss library not found, install it with `pip install mss` command.')

		# enable logging if specified
		self.logging = False
		self.logger = log.getLogger('ScreenGear')
		if logging: self.logging = logging

		# create mss object
		self.mss_object = mss() 
		# create monitor instance for the user-defined monitor
		monitor_instance = None
		if (monitor >= 0):
			monitor_instance = self.mss_object.monitors[monitor]
		else:
			raise ValueError("[ScreenGear:ERROR] :: `monitor` value cannot be negative, Read Docs!")

		# Initialize Queue
		self.queue = None

		#import deque
		from collections import deque
		#define deque and assign it to global var
		self.queue = deque(maxlen=96) #max len 96 to check overflow
		#log it
		if logging: self.logger.debug('Enabling Threaded Queue Mode by default for ScreenGear!') 

		#intiate screen dimension handler
		screen_dims = {}
		#initializing colorspace variable
		self.color_space = None
		try: 
			#reformat proper mss dict and assign to screen dimension handler
			screen_dims = {k.strip(): v for k,v in options.items() if k.strip() in ["top", "left", "width", "height"]}
			# separately handle colorspace value to int conversion
			if not(colorspace is None): 
				self.color_space = capPropId(colorspace.strip())
				if logging: self.logger.debug('Enabling `{}` colorspace for this video stream!'.format(colorspace.strip()))
		except Exception as e:
			# Catch if any error occurred
			if logging: self.logger.exception(str(e))

		# intialize mss capture instance
		self.mss_capture_instance = None
		try:
			# check whether user-defined dimensions are provided
			if screen_dims and len(screen_dims) == 4:
				if logging: self.logger.debug('Setting capture dimensions: {}!'.format(screen_dims)) 
				self.mss_capture_instance = screen_dims #create instance from dimensions
			else:
				self.mss_capture_instance = monitor_instance #otherwise create instance from monitor
			# extract global frame from instance
			self.frame = np.asanyarray(self.mss_object.grab(self.mss_capture_instance))
			if self.threaded_queue_mode:
				#intitialize and append to queue
				self.queue.append(self.frame)
		except ScreenShotError:
			#otherwise catch and log errors
			if logging: self.logger.error(self.mss_object.get_error_details())
			raise ValueError("[ScreenGear:ERROR] :: ScreenShotError caught, Wrong dimensions passed to python-mss, Kindly Refer Docs!")
				
		# thread initialization
		self.thread=None
		# initialize termination flag
		self.terminate = False


	def start(self):
		"""
		start the thread to read frames from the video stream
		"""
		self.thread = Thread(target=self.update, name='ScreenGear', args=())
		self.thread.daemon = True
		self.thread.start()
		return self



	def update(self):
		"""
		Update frames from stream
		"""
		#intialize frame variable
		frame = None
		# keep looping infinitely until the thread is terminated
		while True:
			# if the thread terminate is set, stop the thread
			if self.terminate:
				break

			if self.threaded_queue_mode:
				#check queue buffer for overflow
				if len(self.queue) >= 96:
					#stop iterating if overflowing occurs
					time.sleep(0.000001)
					continue
			try:
				frame = np.asanyarray(self.mss_object.grab(self.mss_capture_instance))
			except ScreenShotError:
				raise RuntimeError(self.mss_object.get_error_details())
				self.terminate = True
				continue
			if frame is None or frame.size == 0:
				#no frames received, then safely exit
				if self.threaded_queue_mode:
					if len(self.queue) == 0: self.terminate = True
				else:
					self.terminate = True

			if not(self.color_space is None):
				# apply colorspace to frames
				color_frame = None
				try:
					if isinstance(self.color_space, int):
						color_frame = cv2.cvtColor(frame, self.color_space)
					else:
						self.color_space = None
						if self.logging: self.logger.debug('Colorspace value {} is not a valid Colorspace!'.format(self.color_space))
				except Exception as e:
					# Catch if any error occurred
					self.color_space = None
					if self.logging:
						self.logger.exception(str(e))
						self.logger.debug('Input Colorspace is not a valid Colorspace!')
				if not(color_frame is None):
					self.frame = color_frame
				else:
					self.frame = frame
			else:
				self.frame = frame
			#append to queue
			if self.threaded_queue_mode: self.queue.append(self.frame)
		# finally release mss resources
		self.mss_object.close()



	def read(self):
		"""
		return the frame
		"""
		while self.threaded_queue_mode:
			if len(self.queue)>0:
				return self.queue.popleft()
		return self.frame



	def stop(self):
		"""
		Terminates the Read process
		"""
		if self.logging: self.logger.debug("Terminating ScreenGear Processes.")
		#terminate Threaded queue mode seperately
		if self.threaded_queue_mode and not(self.queue is None):
			self.queue.clear()
			self.threaded_queue_mode = False
			self.frame = None
		# indicate that the thread should be terminated
		self.terminate = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		if self.thread is not None: 
			self.thread.join()
			#properly handle thread exit
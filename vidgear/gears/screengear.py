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
import numpy as np
from pkg_resources import parse_version
import time



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



class ScreenGear:
	"""
	This Class provides a high-level multi-threaded wrapper around Python-mss library which enables us to easily 
	define an area on the computer screen or an open window to record the live screen high-speed frames and thereby 
	pipeline those frames to any application at expense merely any latency all in a single framework. 
	This Class also supports direct mss parameter manipulations given us flexible control over the input.

	It operates in `Threaded Queue Mode` by default.

	Threaded Queue Mode => Sequentially adds and releases frames to/from deque and handles overflow of this queue. It utilizes 
	Deques that support thread-safe, memory efficient appends and pops from either side of the deque with approximately the 
	same O(1) performance in either direction.  


	:param monitor(int): sets the Positions/Location of monitor where to grab frame from. More information can be found here. 
						/ It default value is 1 (means current monitor will be used).

	:param **options(dict): can be used to pass parameters to ScreenGear Class. 
							/This attribute provides the flexibility to manipulate mss input parameters 
							/directly like the dimensions of the region of the given monitor from where the 
							frames to be grabbed. Checkout VidGear docs for usage details.

	:param (string) colorspace: set the colorspace of the video stream. Its default value is None.

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	"""
	def __init__(self, monitor = 1, colorspace = None, logging = False, **options):

		#intialize threaded queue mode
		self.threaded_queue_mode = True

		try:
			import platform

			if platform.system() == 'Linux':
				from mss.linux import MSS as mss
			elif platform.system() == 'Windows':
				from mss.windows import MSS as mss
			elif platform.system() == 'Darwin':
				from mss.darwin import MSS as mss
			else:
				from mss import mss

			from mss.exception import ScreenShotError

		except ImportError as error:
			raise ImportError('python-mss library not found, install it with `pip install mss` command.')


		self.mss_object = mss() 

		monitor_instance = self.mss_object.monitors[monitor]

		#User-Defined Threaded Queue Mode
		if options:
			if "THREADED_QUEUE_MODE" in options:
				if isinstance(options["THREADED_QUEUE_MODE"],bool):
					self.threaded_queue_mode = options["THREADED_QUEUE_MODE"] #assigsn special parameter to global variable
				del options["THREADED_QUEUE_MODE"] #clean
				#reformat option dict

		self.queue = None
		#intialize deque for video files only 
		if self.threaded_queue_mode:
			#import deque
			from collections import deque
			#define deque and assign it to global var
			self.queue = deque(maxlen=96) #max len 96 to check overflow
			#log it
			if logging:
				print('Enabling Threaded Queue Mode!') 
		else:
			#otherwise disable it
			self.threaded_queue_mode = False

		screen_dims = {}

		#initializing colorspace variable
		self.color_space = None

		try: 
			#reformat proper mss dict
			screen_dims = {k.strip(): v for k,v in options.items() if k.strip() in ["top", "left", "width", "height"]}

			# separately handle colorspace value to int conversion
			if not(colorspace is None):
				self.color_space = capPropId(colorspace.strip())

		except Exception as e:
			# Catch if any error occurred
			if logging:
				print(e)

		self.mss_capture_instance = None

		try:
			if screen_dims and len(screen_dims) == 4:
				self.mss_capture_instance = screen_dims
			else:
				self.mss_capture_instance = monitor_instance

			self.frame = np.asanyarray(self.mss_object.grab(self.mss_capture_instance))

			if self.threaded_queue_mode:
				#intitialize and append to queue
				self.queue.append(self.frame)

		except ScreenShotError:
			raise ValueError("ScreenShotError caught: Wrong dimensions passed to python-mss, Kindly Refer Docs!")
			if logging:
				print(self.mss_object.get_error_details())

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

		frame = None

		# keep looping infinitely until the thread is terminated
		while True:

			# if the thread terminate is set, stop the thread
			if self.terminate:
				break

			if self.threaded_queue_mode:
				#check queue buffer for overflow
				if len(self.queue) < 96:
					pass
				else:
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
					if len(self.queue)>0:
						pass
					else:
						self.terminate = True
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

			#append to queue
			if self.threaded_queue_mode:
				self.queue.append(frame)


		# release mss resources
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



	



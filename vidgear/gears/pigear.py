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

# import the packages
from threading import Thread
from pkg_resources import parse_version
import logging
from .helper import capPropId

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

class PiGear:
	"""
	This class exclusively targets the Raspberry Pi Camera Modules such as OmniVision OV5647 Camera Module and Sony IMX219 Camera Module, 
	to obtain high FPS video by utilizing OpenCV and Picamera libraries with a bit of multithreading. But make sure to enable Raspberry Pi 
	hardware specific settings prior using this class.
	
	:param (tuple) resolution: sets the resolution (width,height). Its default value is (640,480).

	:param (integer) framerate: sets the framerate. Its default value is 25.

	:param (string) colorspace: set colorspace of the video stream. Its default value is None.

	:param (dict) **options: sets parameter supported by PiCamera Class to the input video stream. 
							/ These attribute provides the flexibity to manuplate input raspicam video stream directly. 
							/ Parameters can be passed using this **option, allows you to pass keyworded variable length of arguments to PiGear Class.

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	:param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
					/ This delay is essentially required for camera to warm-up. 
					/ Its default value is 0.

	"""
	def __init__(self, resolution=(640, 480), framerate=25, colorspace = None, logging = False, time_delay = 0, **options):

		try:
			import picamera
			from picamera.array import PiRGBArray
			from picamera import PiCamera
			#print(cv2.__version__)
		except ImportError as error:
			# Output expected ImportErrors.
			raise ImportError('Failed to detect Picamera executables, install it with "pip install picamera" command.')

		# initialize the picamera stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate

		#initialise framerate variable
		self.framerate = framerate

		#initialisation colorspace variable
		self.color_space = None

		#reformat dict
		options = {k.strip(): v for k,v in options.items()}

		try: 
			# apply attributes to source if specified
			for key, value in options.items():
				setattr(self.camera, key, value)

			# seperately handle colorspace value to int conversion
			if not(colorspace is None):
				self.color_space = capPropId(colorspace.strip())

		except Exception as e:
			# Catch if any error occurred
			if logging:
				print(e)

		# enable rgb capture array thread and capture stream
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,format="bgr", use_video_port=True)

		# applying time delay to warmup picamera only if specified
		if time_delay:
			import time
			time.sleep(time_delay)

		#thread initialization
		self.thread = None
		
		#frame initialization
		self.frame = None

		# enable logging if specified
		self.logging = logging

		# intialize termination flag
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
		try:
			for stream in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
				if stream is None:
					if self.logging:
						print('The Camera Module is not working Properly!')
					self.terminate = True
				if self.terminate:
					break
				frame = stream.array
				self.rawCapture.seek(0)
				self.rawCapture.truncate()

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

		except Exception as e:
			if self.logging:
				logging.error(traceback.format_exc())
			self.terminate =True
			pass

		# release resource camera resources
		self.stream.close()
		self.rawCapture.close()
		self.camera.close()

	def read(self):
		"""
		return the frame
		"""
		return self.frame

	def stop(self):
		"""
		Terminates the Read process
		"""
		# indicate that the thread should be terminated
		self.terminate = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		if self.thread is not None: 
			self.thread.join()
			#properly handle thread exit


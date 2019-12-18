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

# import the packages
from threading import Thread
from pkg_resources import parse_version
import sys, time
from .helper import capPropId
import logging as log


try:
	# import OpenCV Binaries
	import cv2

	# check whether OpenCV Binaries are 3.x+
	if parse_version(cv2.__version__) < parse_version('3'):
		raise ImportError('[PiGear:ERROR] :: OpenCV library version >= 3.0 is only supported by this library')

except ImportError as error:
	raise ImportError('[PiGear:ERROR] :: Failed to detect OpenCV executables, install it with `pip3 install opencv-python` command.')



class PiGear:

	"""
	PiGear is similar to CamGear but exclusively made to support various Raspberry Pi Camera Modules 
	(such as OmniVision OV5647 Camera Module and Sony IMX219 Camera Module). To interface with these 
	modules correctly, PiGear provides a flexible multi-threaded wrapper around complete picamera 
	python library and provides us the ability to exploit its various features like `brightness, saturation, sensor_mode`, etc. effortlessly.
	
	:param (tuple) resolution: sets the resolution (width,height). Its default value is (640,480).

	:param (integer) framerate: sets the framerate. Its default value is 25.

	:param (string) colorspace: set colorspace of the video stream. Its default value is None.

	:param (dict) **options: sets parameter supported by PiCamera Class to the input video stream. 
							/ These attribute provides the flexibility to manipulate input raspicam video stream directly. 
							/ Parameters can be passed using this **option, allows you to pass key worded variable length of arguments to PiGear Class.

	:param (boolean) self.logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	:param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
					/ This delay is essentially required for camera to warm-up. 
					/ Its default value is 0.

	"""
	
	def __init__(self, camera_num = 0, resolution = (640, 480), framerate = 30, colorspace = None, logging = False, time_delay = 0, **options):

		try:
			import picamera
			from picamera.array import PiRGBArray
			from picamera import PiCamera
		except Exception as error:
			if isinstance(error, ImportError):
				# Output expected ImportErrors.
				raise ImportError('[PiGear:ERROR] :: Failed to detect Picamera executables, install it with "pip3 install picamera" command.')
			else:
				#Handle any API errors
				raise RuntimeError('[PiGear:ERROR] :: Picamera API failure: {}'.format(error))

		# enable logging if specified
		self.logging = False
		if logging:
			self.logger = log.getLogger('PiGear')
			self.logging = True

		assert (isinstance(framerate, (int, float)) and framerate > 5.0), "[PiGear:ERROR] :: Input framerate value `{}` is a Invalid! Kindly read docs.".format(framerate)
		assert (isinstance(resolution, (tuple, list)) and len(resolution) == 2), "[PiGear:ERROR] :: Input resolution value `{}` is a Invalid! Kindly read docs.".format(resolution)
		if not(isinstance(camera_num, int) and camera_num >= 0): 
			camera_num = 0
			self.logger.warning("Input camera_num value `{}` is invalid, Defaulting to index 0!")

		# initialize the picamera stream at given index
		self.camera = PiCamera(camera_num = camera_num)
		self.camera.resolution = tuple(resolution)
		self.camera.framerate = framerate
		if self.logging: self.logger.debug("Activating Pi camera at index: {} with resolution: {} & framerate: {}".format(camera_num, resolution, framerate))

		#initialize framerate variable
		self.framerate = framerate

		#initializing colorspace variable
		self.color_space = None

		#reformat dict
		options = {k.strip(): v for k,v in options.items()}

		#define timeout variable default value(handles hardware failures)
		self.failure_timeout = 2.0

		#User-Defined parameter
		if options and "HWFAILURE_TIMEOUT" in options:
			#for altering timeout variable manually
			if isinstance(options["HWFAILURE_TIMEOUT"],(int, float)):
				if not(10.0 > options["HWFAILURE_TIMEOUT"] > 1.0): raise ValueError('[PiGear:ERROR] :: `HWFAILURE_TIMEOUT` value can only be between 1.0 ~ 10.0')
				self.failure_timeout = options["HWFAILURE_TIMEOUT"] #assign special parameter
				if self.logging: self.logger.debug("Setting HW Failure Timeout: {} seconds".format(self.failure_timeout))
			del options["HWFAILURE_TIMEOUT"] #clean

		try:
			# apply attributes to source if specified
			for key, value in options.items():
				setattr(self.camera, key, value)

			# separately handle colorspace value to int conversion
			if not(colorspace is None): 
				self.color_space = capPropId(colorspace.strip())
				if self.logging: self.logger.debug('Enabling `{}` colorspace for this video stream!'.format(colorspace.strip()))

		except Exception as e:
			# Catch if any error occurred
			if self.logging: self.logger.exception(str(e))

		# enable rgb capture array thread and capture stream
		self.rawCapture = PiRGBArray(self.camera, size = resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,format="bgr", use_video_port=True)

		#frame variable initialization
		self.frame = None
		try:
			stream = next(self.stream)
			self.frame = stream.array
			self.rawCapture.seek(0)
			self.rawCapture.truncate()
			#render colorspace if defined
			if not(self.frame is None and self.color_space is None): self.frame = cv2.cvtColor(self.frame, self.color_space)
		except Exception as e:
			self.logger.exception(str(e))
			raise RuntimeError('[PiGear:ERROR] :: Camera Module failed to initialize!')

		# applying time delay to warm-up picamera only if specified
		if time_delay: time.sleep(time_delay)

		#thread initialization
		self.thread = None

		#timer thread initialization(Keeps check on frozen thread)
		self._timer = None
		self.t_elasped = 0.0 #records time taken by thread

		# catching thread exceptions
		self.exceptions = None

		# initialize termination flag
		self.terminate = False



	def start(self):
		"""
		start the thread to read frames from the video stream and initiate internal timer
		"""
		#Start frame producer thread
		self.thread = Thread(target=self.update, name='PiGear', args=())
		self.thread.daemon = True
		self.thread.start()

		#Start internal timer thread
		self._timer = Thread(target=self._timeit, name='PiTimer', args=())
		self._timer.daemon = True
		self._timer.start()

		return self



	def _timeit(self):
		"""
		Keep checks on Thread excecution timing
		"""
		#assign current time
		self.t_elasped = time.time()

		#loop until termainated
		while not(self.terminate):
			#check for frozen thread
			if time.time() - self.t_elasped > self.failure_timeout:
				#log failure
				if self.logging: self.logger.critical("Camera Module Disconnected!")
				#prepare for clean exit
				self.exceptions = True
				self.terminate = True #self-terminate



	def update(self):
		"""
		Update frames from stream
		"""
		#keep looping infinitely until the thread is terminated
		while True:

			#check for termination flag
			if self.terminate: break

			try:
				#Try to iterate next frame from generator
				stream = next(self.stream)
			except Exception:
				#catch and save any exceptions
				self.exceptions =  sys.exc_info()
				break #exit

			#update timer
			self.t_elasped = time.time()

			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			frame = stream.array
			self.rawCapture.seek(0)
			self.rawCapture.truncate()

			#apply colorspace if specified
			if not(self.color_space is None):
				# apply colorspace to frames
				color_frame = None
				try:
					if isinstance(self.color_space, int):
						color_frame = cv2.cvtColor(frame, self.color_space)
					else:
						self.color_space = None
						if self.logging: self.logger.debug('Colorspace value `{}` is not a valid colorspace!'.format(self.color_space))
							
				except Exception as e:
					# Catch if any error occurred
					self.color_space = None
					if self.logging:
						self.logger.exception(str(e))
						self.logger.warning('Input colorspace is not a valid Colorspace!')

				if not(color_frame is None):
					self.frame = color_frame
				else:
					self.frame = frame
			else:
				self.frame = frame

		# terminate processes
		if not(self.terminate): self.terminate = True

		# release picamera resources
		self.stream.close()
		self.rawCapture.close()
		self.camera.close()



	def read(self):
		"""
		return the frame
		"""
		#check if there are any thread exceptions
		if not(self.exceptions is None):
			if isinstance(self.exceptions, bool):
				#clear frame
				self.frame = None
				#notify user about hardware failure 
				raise SystemError('[PiGear:ERROR] :: Hardware failure occurred, Kindly reconnect Camera Module and restart your Pi!')
			else:
				#clear frame
				self.frame = None
				# re-raise error for debugging
				error_msg = "[PiGear:ERROR] :: Camera Module API failure occured: {}".format(self.exceptions[1])
				raise RuntimeError(error_msg).with_traceback(self.exceptions[2])

		# return the frame
		return self.frame



	def stop(self):
		"""
		Terminates the Read process
		"""
		if self.logging: self.logger.debug("Terminating PiGear Processes.")

		# make sure that the threads should be terminated
		self.terminate = True

		#stop timer thread
		if not(self._timer is None): self._timer.join()

		#handle camera thread
		if not(self.thread is None):
			#check if hardware failure occured
			if not(self.exceptions is None) and isinstance(self.exceptions, bool):
				# force release picamera resources
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()

				#properly handle thread exit
				self.thread.terminate()
				self.thread.wait() #wait if still process is still processing some information
				self.thread = None
			else:
				#properly handle thread exit
				self.thread.join()
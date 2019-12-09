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
import sys, time
from .helper import capPropId



try:
	# import OpenCV Binaries
	import cv2

	# check whether OpenCV Binaries are 3.x+
	if parse_version(cv2.__version__) < parse_version('3'):
		raise ImportError('[ERROR]: OpenCV library version >= 3.0 is only supported by this library')

except ImportError as error:
	raise ImportError('[ERROR]: Failed to detect OpenCV executables, install it with `pip3 install opencv-python` command.')



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

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

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
				raise ImportError('[ERROR]: Failed to detect Picamera executables, install it with "pip3 install picamera" command.')
			else:
				#Handle any API errors
				raise RuntimeError('[ERROR]: Picamera API failure: {}'.format(error))

		assert (isinstance(framerate, (int, float)) and framerate > 5.0), "[ERROR]: Input framerate value `{}` is a Invalid! Kindly read docs.".format(framerate)
		assert (isinstance(resolution, (tuple, list)) and len(resolution) == 2), "[ERROR]: Input resolution value `{}` is a Invalid! Kindly read docs.".format(resolution)
		if not(isinstance(camera_num, int) and camera_num >= 0): print("[ERROR]: `camera_num` value is invalid, Kindly read docs!")

		# initialize the picamera stream at given index
		self.camera = PiCamera(camera_num = camera_num)
		self.camera.resolution = tuple(resolution)
		self.camera.framerate = framerate
		if logging: print("[LOG]: Activating Pi camera at index: {} with resolution: {} & framerate: {}".format(camera_num, resolution, framerate))

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
				if not(10.0 > options["HWFAILURE_TIMEOUT"] > 1.0): raise ValueError('[ERROR]: `HWFAILURE_TIMEOUT` value can only be between 1.0 ~ 10.0')
				self.failure_timeout = options["HWFAILURE_TIMEOUT"] #assign special parameter
				if logging: print("[LOG]: Setting HW Failure Timeout: {} seconds".format(self.failure_timeout))
			del options["HWFAILURE_TIMEOUT"] #clean

		try:
			# apply attributes to source if specified
			for key, value in options.items():
				setattr(self.camera, key, value)

			# separately handle colorspace value to int conversion
			if not(colorspace is None):
				self.color_space = capPropId(colorspace.strip())

		except Exception as e:
			# Catch if any error occurred
			if logging: print(e)

		# enable rgb capture array thread and capture stream
		self.rawCapture = PiRGBArray(self.camera, size = resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,format="bgr", use_video_port=True)

		#frame variable initialization
		try:
			stream = next(self.stream)
			self.frame = stream.array
			self.rawCapture.seek(0)
			self.rawCapture.truncate()
		except Exception as e:
			print(e)
			raise RuntimeError('[ERROR]: Camera Module failed to initialize!')

		# applying time delay to warm-up picamera only if specified
		if time_delay: time.sleep(time_delay)

		#thread initialization
		self.thread = None

		#timer thread initialization(Keeps check on frozen thread)
		self._timer = None
		self.t_elasped = 0.0 #records time taken by thread

		# enable logging if specified
		self.logging = logging

		# catching thread exceptions
		self.exceptions = None

		# initialize termination flag
		self.terminate = False



	def start(self):
		"""
		start the thread to read frames from the video stream
		"""
		#Start frame producer thread
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()

		#Start internal timer thread
		self._timer = Thread(target=self._timeit, args=())
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
				if self.logging: print("[WARNING]: Camera Module Disconnected!")
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
						if self.logging: print('[LOG]: Colorspace value `{}` is not a valid colorspace!'.format(self.color_space))
							
				except Exception as e:
					# Catch if any error occurred
					self.color_space = None
					if self.logging:
						print(e)
						print('[WARNING]: Input colorspace is not a valid Colorspace!')

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
				raise SystemError('[ERROR]: Hardware failure occurred, Kindly reconnect Camera Module and restart your Pi!')
			else:
				#clear frame
				self.frame = None
				# re-raise error for debugging
				error_msg = "[ERROR]: Camera Module API failure occured: {}".format(self.exceptions[1])
				raise RuntimeError(error_msg).with_traceback(self.exceptions[2])

		# return the frame
		return self.frame



	def stop(self):
		"""
		Terminates the Read process
		"""
		if self.logging: print("[LOG]: Terminating PiGear Process.")

		# make sure that the threads should be terminated
		self.terminate = True

		#stop timer thread
		self._timer.join()

		if self.thread is not None:
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


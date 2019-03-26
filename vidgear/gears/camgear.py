# import the necessary packages
from threading import Thread
from pkg_resources import parse_version
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
	raise ImportError('Failed to detect OpenCV executables, install it with "pip install opencv-python" command.')

class CamGear:
	"""This class targets any common IP or USB Cameras(including Raspberry Pi Compatible), 
	Various Video Files Formats and Network Video Streams(Including Gstreamer Raw Video Capture Pipeline) 
	for obtaining high-speed real-time frames by utilizing OpenCV and multi-threading."""

	def __init__(self, source = 0, logging = False, time_delay = 0, **options):
		# initialize the camera stream and read the first frame

		self.stream = cv2.VideoCapture(source)

		try: 
			# try to apply attributes to source if specified
			for key, value in options.items():
				self.stream.set(self.capPropId(key.strip()),value)
		except Exception as e:
			# Catch if any error occurred
			if logging:
				print(e)

		(self.grabbed, self.frame) = self.stream.read()

		# applying time delay to warm-up webcam only if specified
		if time_delay:
			import time
			time.sleep(time_delay)

		# thread initialization
		self.thread=None

		# initialize termination flag
		self.terminate = False

	def start(self):
		# start the thread to read frames from the video stream
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()
		return self

	def capPropId(self, property):
		#Retrieves the Property's Integer(Actual) value. 
		return getattr(cv2, property)

	def update(self):
		# keep looping infinitely until the thread is terminated
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.terminate:
				break

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

			#check for valid frames
			if not self.grabbed:
				#no frames received, then safely exit
				self.terminate = True
				
		#release resources
		self.stream.release()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be terminate
		self.terminate = True

		# wait until stream resources are released (producer thread might be still grabbing frame)
		if self.thread is not None: 
			self.thread.join()
			#properly handle thread exit

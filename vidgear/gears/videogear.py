# import the necessary packages
from .camgear import CamGear
import time

class VideoGear:
	"""
	This a common class to access any of the vidgear primary classes(i.e. PiGear and CamGear), separated by enablePiCamera flag. 
	This Class accepts different attributes and parameters based on the class that is being accessed currently.
	"""

	def __init__(self, source=0, enablePiCamera = False, resolution=(640, 480), framerate=25, logging = False, time_delay = 0, **options):
		
		if enablePiCamera:
			# only import the pigear module only if required
			from .pigear import PiGear

			# initialize the picamera stream by enabling PiGear Class
			self.stream = PiGear(resolution=resolution, framerate=framerate, logging = logging, time_delay = time_delay, **options)

		else:
			# otherwise, we are using OpenCV so initialize the webcam
			# stream by activating CamGear Class
			self.stream = CamGear(source=source, logging = logging, time_delay = time_delay, **options)

	def start(self):
		# start the threaded video stream
		return self.stream.start()

	def update(self):
		# grab the next frame from the stream
		self.stream.update()

	def read(self):
		# return the current frame
		return self.stream.read()

	def stop(self):
		# stop the thread and release any resources
		self.stream.stop()
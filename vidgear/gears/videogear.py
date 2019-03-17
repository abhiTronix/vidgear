# import the necessary packages
from .camgear import CamGear
import time

class VideoGear:
	def __init__(self, source=0, enablePiCamera = False, resolution=(640, 480), framerate=25, logging = False, time_delay = 0, **options):

		
		if enablePiCamera:
			# only import the pigear module only if required
			from .pigear import PiGear

			# initialize the picamera stream and allow the camera
			# sensor to warmup
			self.stream = PiGear(resolution=resolution, framerate=framerate, logging = logging, time_delay = time_delay, **options)

		# otherwise, we are using OpenCV so initialize the webcam
		# stream
		else:
			self.stream = CamGear(source=source)

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
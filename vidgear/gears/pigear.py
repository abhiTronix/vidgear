# import the packages
from threading import Thread
import logging

try:
	import cv2
	#print(cv2.__version__)
except ImportError as error:
	# Output expected ImportErrors.
	raise ImportError('Failed to detect OpenCV executables, install it with "pip install opencv-python" command.')

class PiGear:

	def __init__(self, resolution=(640, 480), framerate=25, logging = False, time_delay = 0, **options):

		try:
			import picamera
			from picamera import PiRGBArray
			from picamera import PiCamera
			#print(cv2.__version__)
		except ImportError as error:
			# Output expected ImportErrors.
			raise ImportError('Failed to detect Picamera executables, install it with "pip install picamera" command.')

		# initialize the picamera stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate

		# apply attributes to picamera if specified
		for key, value in options.items():
			setattr(self.camera, key, value)

		# enable rgb capture array thread and capture stream
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,format="bgr", use_video_port=True)

		# applying time delay to warmup picamera only if specified
		if time_delay:
			import time
			time.sleep(time_delay)

		#thread intialization
		self.thread = None

		# enable logging if specified
		self.logging = logging

		# intialize termination flag
		self.terminate = False


	def start(self):

		# start the thread to read frames from the video stream
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is terminated
		try:
			for stream in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
				if stream is None:
					self.terminate =True
				if self.terminate:
					break
				self.frame = stream.array
				self.rawCapture.seek(0)
				self.rawCapture.truncate()
		except Exception as e:
			if self.logging:
				logging.error(traceback.format_exc())
			pass

		# release resource camera resources
		self.stream.close()
		self.rawCapture.close()
		self.camera.close()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be terminated
		self.terminate = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		if self.thread is not None: 
			self.thread.join()
			#properly handle thread exit


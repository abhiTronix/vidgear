# import the necessary packages
from threading import Thread

try:
	import cv2
	#print(cv2.__version__)
except ImportError as error:
	# Output expected ImportErrors.
	raise ImportError('Failed to detect OpenCV executables, install it with "pip install opencv-python" command.')

class CamGear:
	def __init__(self, source=0):

		# initialize the camera stream and read the first frame

		self.stream = cv2.VideoCapture(source)

		(self.grabbed, self.frame) = self.stream.read()

		# thread intialization
		self.thread=None

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
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.terminate:
				break

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

			#check for valid frames
			if not self.grabbed:
				#no frames recieved, then safely exit
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

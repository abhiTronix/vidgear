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


def youtube_url_validation(url):
	#convert youtube video url and checks its validity
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.match(youtube_regex, url)

    if youtube_regex_match:
        return youtube_regex_match.group(6)

    return youtube_regex_match


class CamGear:

	"""
	This class targets any common IP or USB Cameras(including Raspberry Pi Compatible), 
	Various Video Files Formats and Network Video Streams(Including Gstreamer Raw Video Capture Pipeline) 
	for obtaining high-speed real-time frames by utilizing OpenCV and multi-threading. It also supports direct Youtube Stream.

	:param source : take the source value. Its default value is 0. Valid Inputs are:

	    - Index(integer): Valid index of the video device.

	    - YouTube Url(string): Youtube URL as input.

	    - Network_Stream_Address(string): Incoming Stream Valid Network address. 

	    - GStreamer (string) videostream Support


	:param (boolean) y_tube: enables YouTube Mode, i.e If enabled class will interpret the given source string as YouTube URL. Its default value is False.

    	:param (dict) **options: sets all properties supported by OpenCV's VideoCapture Class properties to the input video stream in CamGear Class. 
                      / These attribute provides the flexibility to manipulate input webcam video stream directly. 
                      / Parameters can be passed using this **option, allows you to pass keyworded variable length of arguments to CamGear Class.

    	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

    	:param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
    					/ This delay is essentially required for camera to warm-up. 
    					/Its default value is 0.
    """

	def __init__(self, source = 0, y_tube = False, logging = False, time_delay = 0, **options):


		# check if Youtube Mode is ON (True)
		if y_tube:
			#import pafy and parse youtube stream url
			import pafy

			# validate
			url = youtube_url_validation(source)

			if url:
				source_object = pafy.new(url)

				print(source_object.title)
				_source = source_object.getbestvideo("any", ftypestrict=False)
				source = _source.url

			else:
				raise ValueError('Input YouTube Url is invalid!')


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

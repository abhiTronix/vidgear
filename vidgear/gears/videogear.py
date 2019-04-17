# import the necessary packages
from .camgear import CamGear

class VideoGear:
	"""
	This a common secondary class to access any of the vidgear primary classes(i.e. PiGear and CamGear), separated by ```enablePiCamera``` flag. 
	This Class accepts different attributes and parameters based on the class that is being accessed currently. Kindly read `Vidgear Wiki` for more detailed information.


	:param (boolean) enablePiCamera: set this flag to access PiGear or CamGear class respectively. 
									/ This means the if enablePiCamera flag is `True`, PiGear class will be accessed 
									/ and if `False`, the camGear Class will be accessed. Its default value is False.

	:param source : take the source value for CamGear Class. Its default value is 0. Valid Inputs are:

	    - Index(integer): Valid index of the video device.

	    - YouTube Url(string): Youtube URL as input.

	    - Network_Stream_Address(string): Incoming Stream Valid Network address. 

	    - GStreamer (string) videostream Support


	:param (boolean) y_tube: enables YouTube Mode in CamGear Class, i.e If enabled the class will interpret the given source string as YouTube URL. 
							/ Its default value is False.
	
	:param (tuple) resolution: sets the resolution (width,height) in Picamera class. Its default value is (640,480).

	:param (integer) framerate: sets the framerate in Picamera class. Its default value is 25.

    :param (dict) **options: sets parameter supported by PiCamera or Camgear(whichever being accessed) Class to the input video stream. 
    						/ These attribute provides the flexibity to manuplate input raspicam video stream directly. 
    						/ Parameters can be passed using this **option, allows you to pass keyworded variable length of arguments to given Class.

    :param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

    :param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
    					/ This delay is essentially required for camera to warm-up. 
    					/ Its default value is 0.

	"""

	def __init__(self, source=0, enablePiCamera = False, y_tube = False, resolution=(640, 480), framerate=25, logging = False, time_delay = 0, **options):
		
		if enablePiCamera:
			# only import the pigear module only if required
			from .pigear import PiGear

			# initialize the picamera stream by enabling PiGear Class
			self.stream = PiGear(resolution=resolution, framerate=framerate, logging = logging, time_delay = time_delay, **options)

		else:
			# otherwise, we are using OpenCV so initialize the webcam
			# stream by activating CamGear Class
			self.stream = CamGear(source=source, y_tube = y_tube, logging = logging, time_delay = time_delay, **options)

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

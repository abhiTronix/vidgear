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

# import the necessary packages
from .camgear import CamGear
import logging as log


class VideoGear:

	"""
	VideoGear API provides a special internal wrapper around VidGear's exclusive Video Stabilizer class. Furthermore, VideoGear API can provide 
	internal access to both CamGear and PiGear APIs separated by a special flag. Thereby, this API holds the exclusive power for any incoming 
	VideoStream from any source, whether it is live or not, to stabilize it directly with minimum latency and memory requirements.

	VideoGear Specific parameters:
	
		:param (boolean) enablePiCamera: set this flag to access PiGear or CamGear class respectively. 
										/ This means the if enablePiCamera flag is `True`, PiGear class will be accessed 
										/ and if `False`, the camGear Class will be accessed. Its default value is False.

		:param (boolean) stabilize: set this flag to enable access to VidGear's Stabilizer Class. This basically enables(if True) or disables(if False) 
										video stabilization in VidGear. Its default value is False.

		:param (dict) **options: can be used in addition, to pass parameter supported by VidGear's stabilizer class.
								/ Supported dict keys are: 
									- `SMOOTHING_RADIUS` (int) : to alter averaging window size. It handles the quality of stabilization at expense of latency and sudden panning. 
															/ Larger its value, less will be panning, more will be latency and vice-versa. It's default value is 25.
									- `BORDER_SIZE` (int) : to alter output border cropping. It's will crops the border to reduce the black borders from stabilization being too noticeable. 
															/ Larger its value, more will be cropping. It's default value is 0 (i.e. no cropping).			
									- `BORDER_TYPE` (string) : to change the border mode. Valid border types are 'black', 'reflect', 'reflect_101', 'replicate' and 'wrap'. It's default value is 'black'
		
		:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.
	
	CamGear Specific supported parameters for VideoGear:

		:param source : take the source value for CamGear Class. Its default value is 0. Valid Inputs are:
			- Index(integer): Valid index of the video device.
			- YouTube Url(string): Youtube URL as input.
			- Network_Stream_Address(string): Incoming Stream Valid Network address. 
			- GStreamer (string) videostream Support
		:param (boolean) y_tube: enables YouTube Mode in CamGear Class, i.e If enabled the class will interpret the given source string as YouTube URL. 
								/ Its default value is False.
		:param (int) backend: set the backend of the video stream (if specified). Its default value is 0.


	PiGear Specific supported parameters for VideoGear:
	
		:param (tuple) resolution: sets the resolution (width,height) in Picamera class. Its default value is (640,480).
		:param (integer) framerate: sets the framerate in Picamera class. Its default value is 25.


	Common parameters for CamGear and PiGear: 
		:param (dict) **options: sets parameter supported by PiCamera or Camgear (whichever being accessed) Class to the input video stream. 
								/ These attribute provides the flexibity to manuplate input raspicam video stream directly. 
								/ Parameters can be passed using this **option, allows you to pass keyworded variable length of arguments to given Class.
		:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.
		:param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
							/ This delay is essentially required for camera to warm-up. 
							/ Its default value is 0.
	"""

	def __init__(self, enablePiCamera = False, stabilize = False, source = 0, y_tube = False, backend = 0, colorspace = None, resolution = (640, 480), framerate = 25, logging = False, time_delay = 0, **options):
		
		#initialize stabilizer
		self.__stablization_mode = stabilize

		# enable logging if specified
		self.__logging = False
		self.__logger = log.getLogger('VideoGear')
		if logging: self.__logging = logging

		if self.__stablization_mode:
			from .stabilizer import Stabilizer
			s_radius, border_size, border_type, crop_n_zoom = (25, 0, 'black', False) #defaults
			if options:
				if "SMOOTHING_RADIUS" in options:
					if isinstance(options["SMOOTHING_RADIUS"],int):
						s_radius = options["SMOOTHING_RADIUS"] #assigsn special parameter to global variable
					del options["SMOOTHING_RADIUS"] #clean
				if "BORDER_SIZE" in options:
					if isinstance(options["BORDER_SIZE"],int):
						border_size = options["BORDER_SIZE"] #assigsn special parameter
					del options["BORDER_SIZE"] #clean
				if "BORDER_TYPE" in options:
					if isinstance(options["BORDER_TYPE"],str):
						border_type = options["BORDER_TYPE"] #assigsn special parameter
					del options["BORDER_TYPE"] #clean
				if "CROP_N_ZOOM" in options:
					if isinstance(options["CROP_N_ZOOM"],bool):
						crop_n_zoom = options["CROP_N_ZOOM"] #assigsn special parameter
					del options["CROP_N_ZOOM"] #clean
			self.__stabilizer_obj = Stabilizer(smoothing_radius = s_radius, border_type = border_type, border_size = border_size, crop_n_zoom = crop_n_zoom, logging = logging)
			if self.__logging: self.__logger.debug('Enabling Stablization Mode for the current video source!') #log info

		if enablePiCamera:
			# only import the pigear module only if required
			from .pigear import PiGear

			# initialize the picamera stream by enabling PiGear Class
			self.stream = PiGear(resolution = resolution, framerate = framerate, colorspace = colorspace, logging = logging, time_delay = time_delay, **options)
		else:
			# otherwise, we are using OpenCV so initialize the webcam
			# stream by activating CamGear Class
			self.stream = CamGear(source = source, y_tube = y_tube, backend = backend, colorspace = colorspace, logging = logging, time_delay = time_delay, **options)

		#initialize framerate variable
		self.framerate = self.stream.framerate


	def start(self):
		# start the threaded video stream
		self.stream.start()
		return self


	def read(self):
		# return the current frame
		while self.__stablization_mode:
			frame = self.stream.read()
			if frame is None:
				break
			frame_stab = self.__stabilizer_obj.stabilize(frame)
			if not(frame_stab is None):
				return frame_stab
		return self.stream.read()


	def stop(self):
		if self.__logging: self.__logger.debug("Terminating VideoGear.")
		# stop the thread and release any resources
		self.stream.stop()
		#clean queue
		if self.__stablization_mode: self.__stabilizer_obj.clean()
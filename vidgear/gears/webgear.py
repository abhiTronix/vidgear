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
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.routing import Mount
from starlette.routing import Route

from .videogear import VideoGear
from .helper import reducer
from .helper import logger_handler
from .helper import check_python_version
from .helper import generate_webdata
from collections import deque

import logging as log
import os, cv2, asyncio, sys


#define logger
logger = log.getLogger('WebGear')
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class WebGear:

	"""
	WebGear is a powerful ASGI Video-streamer API, that is built upon Starlette - a lightweight ASGI framework/toolkit, which is ideal 
	for building high-performance asyncio services.

	WebGear API provides a flexible but robust asyncio wrapper around Starlette ASGI application and can easily access its various components 
	independently. Thereby providing it the ability to interact with the Starlette's ecosystem of shared middleware and mountable applications 
	& seamless access to its various Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc.

	WebGear acts as robust Live Video Streaming Server that can stream live video frames to any web browser on a network in real-time. 
	It addition to this, WebGear provides a special internal wrapper around VideoGear API, which itself provides internal access to both CamGear and PiGear 
	APIs thereby granting it exclusive power for streaming frames incoming from any device/source. Also on the plus side, since WebGear has access 
	to all functions of VideoGear API, therefore it can stabilize video frames even while streaming live.
	
	"""

	def __init__(self, enablePiCamera = False, stabilize = False, source = 0, camera_num = 0, y_tube = False, backend = 0, colorspace = None, resolution = (640, 480), framerate = 25, logging = False, time_delay = 0, **options):
		
		#assert if python in-use is valid
		assert check_python_version() >= (3,6), "[WebGear:ERROR] :: Invalid python version. WebGear API only supports python 3.6+ versions!"

		#reformat dict
		options = {k.lower().strip(): v for k,v in options.items()}

		#initialize global params
		self.__jpeg_quality = 90 #90% quality
		self.__jpeg_optimize = 0 #optimization off
		self.__jpeg_progressive=0 #jpeg will be baseline instead
		self.__frame_size_reduction = 20 #20% reduction
		self.__logging = logging

		custom_data_location = '' #path to save data-files to custom location
		data_path = '' #path to WebGear data-files
		overwrite_default=False

		# assign values to global variables if specified and valid
		if options:
			if 'frame_size_reduction' in options:
				value = options["frame_size_reduction"]
				if isinstance(value, (int, float)) and value >= 0 and value <= 90:
					self.__frame_size_reduction = value
				else: 
					logger.warning("Skipped invalid `frame_size_reduction` value!")
				del options['frame_size_reduction'] #clean
			if 'frame_jpeg_quality' in options:
				value = options["frame_jpeg_quality"]
				if isinstance(value, (int, float)) and value >= 10 and value <= 95:
					self.__jpeg_quality = int(value)
				else: 
					logger.warning("Skipped invalid `frame_jpeg_quality` value!")
				del options['frame_jpeg_quality'] #clean
			if 'frame_jpeg_optimize' in options:
				value = options["frame_jpeg_optimize"]
				if isinstance(value, bool): 
					self.__jpeg_optimize = int(value)
				else: 
					logger.warning("Skipped invalid `frame_jpeg_optimize` value!")
				del options['frame_jpeg_optimize'] #clean
			if 'frame_jpeg_progressive' in options:
				value = options["frame_jpeg_progressive"]
				if isinstance(value, bool):
					self.__jpeg_progressive = int(value)
				else: 
					logger.warning("Skipped invalid `frame_jpeg_progressive` value!")
				del options['frame_jpeg_progressive'] #clean
			if 'custom_data_location' in options:
				value = options["custom_data_location"]
				if isinstance(value,str) in options:
					try:
						assert os.access(value, os.W_OK), "[WebGear:ERROR] :: Permission Denied!, cannot write WebGear data-files to '{}' directory!".format(value)
						assert not(os.path.isfile(value)), "[WebGear:ERROR] :: `custom_data_location` value must be the path to a directory and not to a file!"
						custom_data_location = os.path.abspath(value)
					except Exception as e:
						logger.exception(str(e))
				else:
					logger.warning("Skipped invalid `custom_data_location` value!")
				del options['custom_data_location'] #clean
			if 'overwrite_default_files' in options:
				value = options["overwrite_default_files"]
				if isinstance(value, bool):
					overwrite_default = value
				else:
					logger.warning("Skipped invalid `overwrite_default_files` value!")
				del options['overwrite_default_files'] #clean

		#define stream with necessary params
		self.stream = VideoGear(enablePiCamera = enablePiCamera, stabilize = stabilize, source = source, camera_num = camera_num, y_tube = y_tube, backend = backend, colorspace = colorspace, resolution = resolution, framerate = framerate, logging = logging, time_delay = time_delay, **options)

		#check if custom certificates path is specified
		if custom_data_location:
			if os.path.isdir(custom_data_location): #custom certificate location must be a directory
				data_path = generate_webdata(custom_data_location, overwrite_default = overwrite_default, logging = logging)
			else:
				raise ValueError("[WebGear:ERROR] :: Invalid `custom_data_location` value!")
		else:
			# otherwise auto-encapsulation for class functions and variablesgenerate suitable path
			from os.path import expanduser
			data_path = generate_webdata(os.path.join(expanduser("~"),".vidgear"), overwrite_default = overwrite_default, logging = logging)
		
		#log it
		if self.__logging: logger.debug('`{}` is the default location for saving WebGear data-files.'.format(data_path))
		if self.__logging: logger.debug('Setting params:: Size Reduction:{}%, JPEG quality:{}%, JPEG optimizations:{}, JPEG progressive:{}'.format(self.__frame_size_reduction, self.__jpeg_quality, bool(self.__jpeg_optimize), bool(self.__jpeg_progressive)))

		#define Jinja2 templates handler
		self.__templates = Jinja2Templates(directory='{}/templates'.format(data_path))

		#define custom exception handlers
		self.__exception_handlers = {404: self.__not_found,
									500: self.__server_error}
		#define routing tables
		self.routes = [Route('/', endpoint=self.__homepage),
						Route('/video', endpoint=self.__video),
						Mount('/static', app=StaticFiles(directory='{}/static'.format(data_path)), name="static")]
		#copy original routing tables for verfication
		self.__rt_org_copy = self.routes[:]
		#keeps check if producer loop should be running
		self.__isrunning = True



	def __call__(self):
		"""
		Implements custom callable method
		"""
		#validate routing tables
		assert not(self.routes is None), "Routing tables are NoneType!"
		if not isinstance(self.routes, list) or not all(x in self.routes for x in self.__rt_org_copy): raise RuntimeError("Routing tables are not valid!")
		#initate stream
		if self.__logging: logger.debug('Initiating Video Streaming.')
		self.stream.start()
		#return Starlette application
		if self.__logging: logger.debug('Running Starlette application.')
		return Starlette(debug = (True if self.__logging else False), routes=self.routes, exception_handlers=self.__exception_handlers, on_shutdown=[self.__shutdown])



	async def __producer(self):
		"""
		Implements async frame producer.
		"""
		# loop over frames
		while self.__isrunning:
			#read frame
			frame = self.stream.read()
			#break if NoneType
			if frame is None: break
			#reducer frames size if specified
			if self.__frame_size_reduction: frame = reducer(frame, percentage = self.__frame_size_reduction)
			#handle JPEG encoding
			encodedImage = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.__jpeg_quality, cv2.IMWRITE_JPEG_PROGRESSIVE, self.__jpeg_progressive,cv2.IMWRITE_JPEG_OPTIMIZE, self.__jpeg_optimize])[1].tobytes()
			#yield frame in byte format
			yield  (b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+encodedImage+b'\r\n')
			await asyncio.sleep(0.01)



	async def __video(self, scope):
		"""
		Return a async video streaming response.
		"""
		assert scope['type'] == 'http'
		return StreamingResponse(self.__producer(), media_type='multipart/x-mixed-replace; boundary=frame')



	async def __homepage(self, request):
		"""
		Return an HTML index page.
		"""
		return self.__templates.TemplateResponse('index.html', {'request': request})



	async def __not_found(self, request, exc):
		"""
		Return an HTML 404 page.
		"""
		return self.__templates.TemplateResponse('404.html', {'request': request}, status_code=404)



	async def __server_error(self, request, exc):
		"""
		Return an HTML 500 page.
		"""
		return self.__templates.TemplateResponse('500.html', {'request': request}, status_code=500)



	def __shutdown(self):
		"""
		Implements a callables to run on application shutdown
		"""
		if self.__logging: logger.debug('Closing Video Streaming.')
		#stops frame producer
		self.__isrunning = False
		#stops VideoGear stream
		self.stream.stop()
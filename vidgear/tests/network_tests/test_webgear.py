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

import tempfile, asyncio
import os, pytest
import logging as log

from vidgear.gears import WebGear
from vidgear.gears.helper import logger_handler
from starlette.testclient import TestClient
from starlette.responses import PlainTextResponse
from starlette.routing import Route

logger = log.getLogger('Test_webgear')
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def return_testvideo_path():
	"""
	returns Test Video path
	"""
	path = '/home/abhishek/big_buck_bunny_720p_1mb.mp4'
	return os.path.abspath(path)


def hello_webpage(request):
	"""
	returns PlainTextResponse callback for hello world webpage
	"""
	return PlainTextResponse('Hello, world!')



test_data = [ (return_testvideo_path(), False, True, None, 0), 
('https://youtu.be/YqeW9_5kURI', True, False, None, 0),
(return_testvideo_path(), False, False, 'COLOR_BGR2HSV', 10)]

@pytest.mark.parametrize('source, y_tube, stabilize, colorspace, time_delay', test_data)
def test_webgear_class(source, y_tube, stabilize, colorspace, time_delay):
	"""
	Test for various WebGear API parameters
	"""
	try:
		web = WebGear(source = source, y_tube = y_tube, stabilize = stabilize, colorspace = colorspace, time_delay = time_delay, logging = True)
		client = TestClient(web(),raise_server_exceptions=True)
		response = client.get('/')
		assert response.status_code == 200
		response_404 = client.get('/test')
		assert response_404.status_code == 404
		web.shutdown()
	except Exception as e:
		pytest.fail(str(e))



test_data = [{"frame_size_reduction": 47, "frame_jpeg_quality": 88, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False},
{"overwrite_default_files": True},
{"custom_data_location": tempfile.gettempdir()}]

@pytest.mark.parametrize('options', test_data)
def test_webgear_options(options):
	"""
	Test for various WebGear API internal options
	"""
	try:
		web = WebGear(source = return_testvideo_path(),logging = True, **options)
		client = TestClient(web(),raise_server_exceptions=True)
		response = client.get('/')
		assert response.status_code == 200
		response_video = client.get('/video')
		assert response_video.status_code == 200
		web.shutdown()
	except Exception as e:
		pytest.fail(str(e))



def test_webgear_routes():
	"""
	Test for WebGear API's custom routes
	"""
	try:
		#add various performance tweaks as usual
		options = {"frame_size_reduction": 40, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}
		#initialize WebGear app  
		web = WebGear(source = return_testvideo_path(),logging = True, **options) #enable source i.e. `test.mp4` and enable `logging` for debugging 

		#modify route to point our rendered webpage 
		web.routes.append(Route('/hello', endpoint=hello_webpage)) 

		#test
		client = TestClient(web(),raise_server_exceptions=True)
		response = client.get('/')
		assert response.status_code == 200
		response_hello = client.get('/hello')
		assert response_hello.status_code == 200
		web.shutdown()
	except Exception as e:
		pytest.fail(str(e))

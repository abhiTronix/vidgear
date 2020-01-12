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

from vidgear.gears import ScreenGear
from mss.exception import ScreenShotError
from vidgear.gears.helper import logger_handler
import pytest, platform
import logging as log

logger = log.getLogger('Test_screengear')
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def test_screengear():
	"""
	Tests ScreenGear's playback capabilities with custom defined dimensions -> passes if fails with ScreenShotError
	"""
	try:
		# define dimensions of screen w.r.t to given monitor to be captured
		options = {'top': 40, 'left': 0, 'width': 100, 'height': 100} 
		#Open Live Screencast on current monitor 
		stream = ScreenGear(monitor=1, logging=True, colorspace = 'COLOR_BGR2GRAY', **options).start() 
		#playback
		i = 0
		while (i<20):
			frame = stream.read()
			if frame is None: break
			if (i == 10): stream.color_space = "red" #invalid colorspace value
			i+=1
		#clean resources
		stream.stop()
	except Exception as e:
		if platform.system() == 'Linux' or platform.system() == 'Windows':
			logger.exception(e)
		else:
			pytest.fail(str(e))
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

import pytest, os, tempfile
from vidgear.gears import VideoGear
from vidgear.gears.helper import logger_handler
import logging as log

logger = log.getLogger('Test_videogear')
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def return_testvideo_path():
	"""
	returns Test video path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
	return os.path.abspath(path)



def test_PiGear_import():
	"""
	Testing VideoGear Import -> assign to fail when PiGear class is imported
	"""
	with pytest.raises(ImportError):
		stream = VideoGear(enablePiCamera = True, logging = True).start()
		stream.stop()



#Video credit: http://www.liushuaicheng.org/CVPR2014/index.html
test_data = [ ('https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/example4_train_input.mp4', {'SMOOTHING_RADIUS': 5, 'BORDER_SIZE': 10, 'BORDER_TYPE': 'replicate', 'CROP_N_ZOOM': True}), 
			(return_testvideo_path(), {'BORDER_TYPE':'im_wrong'})]

@pytest.mark.parametrize('source, options', test_data)
def test_video_stablization(source, options):
	"""
	Testing VideoGear's Video Stablization playback capabilities 
	"""
	try:
		#open stream
		stab_stream = VideoGear(source = source, stabilize = True, logging = True, **options).start()
		framerate = stab_stream.framerate
		#playback
		while True:
			frame = stab_stream.read() #read stablized frames
			if frame is None:
				break
		#clean resources
		stab_stream.stop()
		logger.debug('Input Framerate: {}'.format(framerate))
		assert framerate>0
	except Exception as e:
		pytest.fail(str(e))
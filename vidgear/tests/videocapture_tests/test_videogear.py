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



def test_CamGear_import():
	"""
	Testing VideoGear Import -> passed if CamGear Class is Imported sucessfully 
	and returns a valid framerate
	"""
	try:
		options = {'THREADED_QUEUE_MODE':False}
		output_stream = VideoGear(source = return_testvideo_path(), logging=True, **options).start()
		framerate = output_stream.framerate
		output_stream.stop()
		print('[LOG] Input Framerate: {}'.format(framerate))
		assert framerate>0
	except Exception as e:
		pytest.fail(str(e))



def test_video_stablization():
	"""
	Testing VideoGear's Video Stablization playback capabilities 
	"""
	try:
		#Video credit: http://www.liushuaicheng.org/CVPR2014/index.html
		Url = 'https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/example4_train_input.mp4'
		#define params
		options = {'SMOOTHING_RADIUS': 5, 'BORDER_SIZE': 10, 'BORDER_TYPE': 'replicate', 'CROP_N_ZOOM': True}
		#open stream
		stab_stream = VideoGear(source = Url, stabilize = True, logging = True, **options).start()
		#playback
		while True:
			frame = stab_stream.read() #read stablized frames
			if frame is None:
				break
		#clean resources
		stab_stream.stop()
	except Exception as e:
		pytest.fail(str(e))

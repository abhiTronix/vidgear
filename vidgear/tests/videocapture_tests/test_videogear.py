"""
============================================
vidgear library code is placed under the MIT license
Copyright (c) 2019 Abhishek Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
"""

#Video credit: http://www.liushuaicheng.org/CVPR2014/index.html

import pytest, os
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
		Url = 'http://www.liushuaicheng.org/CVPR2014/data/example4_train_input.avi'
		#define params
		options = {'SMOOTHING_RADIUS': 5, 'BORDER_SIZE': 0, 'BORDER_TYPE': 'replicate'}
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

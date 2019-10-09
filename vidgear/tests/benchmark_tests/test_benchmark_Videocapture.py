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

import os
import cv2
import pytest
import tempfile
from vidgear.gears import CamGear
from .fps import FPS



def return_testvideo_path():
	"""
	return Test Video Data path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny.mp4'.format(tempfile.gettempdir())
	return os.path.abspath(path)



def Videocapture_withCV(path):
	"""
	Function to benchmark OpenCV video playback 
	"""
	stream = cv2.VideoCapture(path)
	fps_CV = FPS().start()
	while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
		fps_CV.update()
	fps_CV.stop()
	stream.release()
	print("OpenCV")
	print("[LOG] total elasped time: {:.2f}".format(fps_CV.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps_CV.fps()))



def Videocapture_withVidGear(path):
	"""
	Function to benchmark VidGear multi-threaded video playback 
	"""
	options = {'THREADED_QUEUE_MODE':False}
	stream = CamGear(source=path, **options).start()
	fps_Vid = FPS().start()
	while True:
		frame = stream.read()
		if frame is None:
			break
		fps_Vid.update()
	fps_Vid.stop()
	stream.stop()
	print("VidGear")
	print("[LOG] total elasped time: {:.2f}".format(fps_Vid.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps_Vid.fps()))


@pytest.mark.xfail(raises=RuntimeError)
def test_benchmark_videocapture():
	"""
	Benchmarking OpenCV playback against VidGear playback (in FPS)
	"""
	try:
		Videocapture_withCV(return_testvideo_path())
		Videocapture_withVidGear(return_testvideo_path())
	except Exception as e:
		raise RuntimeError(e)

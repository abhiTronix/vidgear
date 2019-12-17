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

import os
import cv2
import pytest
import tempfile
from vidgear.gears import CamGear
from .fps import FPS



def return_testvideo_path():
	"""
	returns Test Video path
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
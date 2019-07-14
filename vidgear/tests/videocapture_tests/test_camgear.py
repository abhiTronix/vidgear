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


import youtube_dl
import cv2
import os, sys, time
import numpy as np
import pytest
from vidgear.gears import CamGear
from numpy.testing import assert_equal
from vidgear.gears.helper import capPropId



def return_youtubevideo_params(url):
	"""
	return Youtube Video parameters(FPS, dimensions) directly using Youtube-dl
	"""
	ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s','noplaylist': True,'quiet': True,'format': 'bestvideo'})
	with ydl:
		result = ydl.extract_info(url,download=False) # We just want to extract the info
	return (int(result['width']),int(result['height']),float(result['fps']))



def return_testvideo_path():
	"""
	return Test Video Data path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)



def return_total_frame_count():
	"""
	simply counts the total frames in a given video
	"""
	stream = cv2.VideoCapture(return_testvideo_path())
	num_cv=0
	while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			print(num_cv)
			break
		num_cv += 1
	stream.release()
	return num_cv



@pytest.mark.xfail(raises=AssertionError)
def test_threaded_queue_mode():
	"""
	Test for New Thread Queue Mode in CamGear Class
	"""
	actual_frame_num = return_total_frame_count()

	stream_camgear = CamGear(source=return_testvideo_path(), logging=True).start() #start stream on CamGear
	camgear_frames_num = 0
	while True:
		frame = stream_camgear.read()
		if frame is None:
			print(camgear_frames_num)
			break
		
		time.sleep(0.2) #dummy computational task

		camgear_frames_num += 1
	stream_camgear.stop()

	assert camgear_frames_num == actual_frame_num



@pytest.mark.xfail(raises=AssertionError)
def test_youtube_playback():
	"""
	Testing Youtube Video Playback capabilities of VidGear
	"""
	if os.name != 'nt':
		Url = 'https://youtu.be/YqeW9_5kURI'
		result = True
		errored = False #keep watch if youtube streaming not successful
		try:
			true_video_param = return_youtubevideo_params(Url)
			options = {'THREADED_QUEUE_MODE':False}
			stream = CamGear(source=Url, y_tube = True, logging=True, **options).start() # YouTube Video URL as input
			height = 0
			width = 0
			fps = 0
			while True:
				frame = stream.read()
				if frame is None:
					break
				if height == 0 or width == 0:
					fps = stream.framerate
					height,width = frame.shape[:2]
			print('WIDTH: {} HEIGHT: {} FPS: {}'.format(true_video_param[0],true_video_param[1],true_video_param[2]))
			print('WIDTH: {} HEIGHT: {} FPS: {}'.format(width,height,fps))
		except Exception as error:
			print(error)
			errored = True

		if not errored:
			assert true_video_param[0] == width and true_video_param[1] == height and true_video_param[2] == fps
		else:
			print('YouTube playback Test is skipped due to above error!')

	else:
		print('YouTube playback Test is skipped due to bug with Appveyor on Windows builds!')



def test_network_playback():
	"""
	Testing Direct Network Video Playback capabilities of VidGear(with rtsp streaming)
	"""	
	Url = 'rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov'
	try:
		options = {'THREADED_QUEUE_MODE':False}
		output_stream = CamGear(source = Url, **options).start()
		i = 0
		Output_data = []
		while i<10:
			frame = output_stream.read()
			if frame is None:
				break
			Output_data.append(frame)
			i+=1
		output_stream.stop()
		print('Output data shape:', np.array(Output_data).shape)
	except Exception as e:
		pytest.fail(str(e))












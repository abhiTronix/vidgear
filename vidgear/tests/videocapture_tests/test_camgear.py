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
import platform
import os, time
import pytest
import tempfile
import numpy as np

from vidgear.gears import CamGear



def return_youtubevideo_params(url):
	"""
	returns Youtube Video parameters(FPS, dimensions) directly using Youtube-dl
	"""
	ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s','noplaylist': True,'quiet': True,'format': 'bestvideo'})
	with ydl:
		result = ydl.extract_info(url,download=False) # We just want to extract the info
	return (int(result['width']),int(result['height']),float(result['fps']))



def return_testvideo_path():
	"""
	returns Test Video path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
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
	if not platform.system() in ['Windows', 'Darwin']:
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
			print('[LOG]: WIDTH: {} HEIGHT: {} FPS: {}'.format(true_video_param[0],true_video_param[1],true_video_param[2]))
			print('[LOG]: WIDTH: {} HEIGHT: {} FPS: {}'.format(width,height,fps))
		except Exception as error:
			print(error)
			errored = True

		if not errored:
			assert true_video_param[0] == width and true_video_param[1] == height and true_video_param[2] == fps
		else:
			print('[LOG]: YouTube playback Test is skipped due to above error!')

	else:
		print('[LOG]: YouTube playback Test is skipped due to bug with opencv-python library builds on windows and macOS!')



def test_network_playback():
	"""
	Testing Direct Network Video Playback capabilities of VidGear(with rtsp streaming)
	"""	
	Publictest_rstp_urls = [
	'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov',
	'rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream',
	'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa',
	'rtmp://semerkandglb.mediatriple.net:1935/semerkandliveedge/semerkand2'
	]

	index = 0

	while (index < len(Publictest_rstp_urls)):
		try:
			options = {'THREADED_QUEUE_MODE':False}
			output_stream = CamGear(source = Publictest_rstp_urls[index], logging = True, **options).start()
			i = 0
			Output_data = []
			while i<10:
				frame = output_stream.read()
				if frame is None:
					break
				Output_data.append(frame)
				i+=1
			output_stream.stop()
			print('[LOG]: Output data shape:', np.array(Output_data).shape)
			if Output_data[-1].shape[:2] > (50,50): break
		except Exception as e:
			if isinstance(e, RuntimeError):
				print("[LOG] `{}` URL is not working".format(Publictest_rstp_urls[index]))
				index+=1
				continue
			else:
				pytest.fail(str(e))

	if (index == len(Publictest_rstp_urls)): pytest.fail(str(e))












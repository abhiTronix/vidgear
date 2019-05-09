import youtube_dl
import cv2
import os, sys
import numpy as np
import pytest
from vidgear.gears import CamGear
from numpy.testing import assert_equal
from vidgear.gears.helper import capPropId

def return_testimage_dir():
	path = '{}/Downloads/Test_images'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)


def return_youtubevideo_params(url):
	ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s','noplaylist': True,'quiet': True,'format': 'bestvideo'})
	with ydl:
		result = ydl.extract_info(url,download=False) # We just want to extract the info
	return (int(result['width']),int(result['height']),float(result['fps']))


def return_testvideo_path():
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)

def prepare_testframes(conversion = ''):
	stream = cv2.VideoCapture(return_testvideo_path())
	j=0
	while True:
		(grabbed, frame) = stream.read()
		# read frames
		# check if frame empty
		if not grabbed:
			#if True break the infinite loop
			break
		if conversion:
			frame = cv2.cvtColor(frame, capPropId(conversion))
		cv2.imwrite('{}.png'.format(return_testimage_dir()+'/'+str(j)),frame)
		j+=1
	stream.release()

@pytest.mark.xfail(raises=AssertionError)
def test_youtube_playback():
	Url = 'https://youtu.be/dQw4w9WgXcQ'
	result = True
	true_video_param = return_youtubevideo_params(Url)
	stream = CamGear(source=Url, y_tube = True,  time_delay=2, logging=True).start() # YouTube Video URL as input
	fps = stream.framerate
	height = 0
	width = 0 
	while True:
		frame = stream.read()
		if frame is None:
			result = False
			break
		if height == 0 or width == 0:
			height,width = frame.shape[:2]
			break
	print('WIDTH: {} HEIGHT: {} FPS: {}'.format(true_video_param[0],true_video_param[1],true_video_param[2]))
	print('WIDTH: {} HEIGHT: {} FPS: {}'.format(width,height,fps))
	if result:
		assert true_video_param[0] == width and true_video_param[1] == height and true_video_param[2] == fps
"""
@pytest.mark.xfail(raises=AssertionError)
def test_video_playback():
	prepare_testframes()
	output_stream = CamGear(source = return_testvideo_path()).start()
	test_images = [x for x in sorted(os.listdir(return_testimage_dir()))]
	i = 0 
	while True:
		frame = output_stream.read()
		if frame is None:
			break
		test_img = cv2.imread(test_images[i], cv2.IMREAD_UNCHANGED)
		assert_equal(frame, test_img, err_msg='Test Failed!')
		os.remove(test_images[i])
		i+=1
	output_stream.stop()

@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize('conversion', ['COLOR_BGR2GRAY', 'COLOR_BGR2YUV', 'COLOR_BGR2HSV'])
def test_color_manuplation(conversion):
	prepare_testframes(conversion)
	output_stream = CamGear(source = return_testvideo_path(), colorspace = conversion).start()
	test_images = [x for x in sorted(os.listdir(return_testimage_dir()))]
	i = 0 
	while True:
		frame = output_stream.read()
		if frame is None:
			break
		test_img = cv2.imread(test_images[i], cv2.IMREAD_UNCHANGED)
		assert_equal(frame, test_img, err_msg='Test Failed!')
		os.remove(test_images[i])
		i+=1
	output_stream.stop()
"""
def test_network_playback():
	Url = 'rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov'
	try:
		output_stream = CamGear(source = Url).start()
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
		if result:
			pytest.fail(str(e))












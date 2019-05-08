import youtube_dl
import cv2
import os, sys
import numpy as np
import pytest
from vidgear.gears import VideoGear
from numpy.testing import assert_equal
from vidgear.gears.helper import capPropId

def return_testimage_dir():
	path = '{}/download/Test_images'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)


def return_youtubevideo_params(url):
	ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s','noplaylist': True,'quiet': True,'format': 'bestvideo'})

	with ydl:
		result = ydl.extract_info(url,download=False) # We just want to extract the info

	return (int(result['width']),int(result['height']),float(result['fps']))


def return_testvideo_path():
	path = '{}/download/Test_videos/BigBuckBunny_4sec.mp4'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
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


def_test_PiGear_import():
	with pytest.raises(ImportError):
		stream = VideoGear(enablePiCamera = True, logging = True).start() # define various attributes and start the stream


@pytest.mark.xfail(raises=AssertionError)
def test_youtube_playback():
	Url = 'https://youtu.be/dQw4w9WgXcQ'
	true_video_param = return_youtubevideo_params(Url)
	stream = VideoGear(source=Url, y_tube = True,  time_delay=1, logging=True).start() # YouTube Video URL as input
	fps = stream.framerate
	width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
	assert true_video_param[0] == width and true_video_param[1] == height and true_video_param[2] == fps

@pytest.mark.xfail(raises=AssertionError)
def test_video_playback():
	prepare_testframes()
	output_stream = VideoGear(source = return_testvideo_path()).start()
	test_images = [x for x in sorted(os.listdir(return_testimage_dir()))]
	i = 0 
	while True:
		frame = output_stream.read()
		if frame is None:
			break
		test_img = cv2.imread(test_images[i], cv2.IMREAD_UNCHANGED)
		assert_equal(frame == test_img)
		os.remove(test_images[i])
		i+=1
	output_stream.stop()


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize('conversion', ['COLOR_BGR2GRAY', 'COLOR_BGR2YUV', 'COLOR_BGR2HSV'])
def test_color_manuplation(conversion):
	prepare_testframes(conversion)
	output_stream = VideoGear(source = return_testvideo_path(), colorspace = conversion).start()
	test_images = [x for x in sorted(os.listdir(return_testimage_dir()))]
	i = 0 
	while True:
		frame = output_stream.read()
		if frame is None:
			break
		test_img = cv2.imread(test_images[i], cv2.IMREAD_UNCHANGED)
		assert_equal(frame == test_img)
		os.remove(test_images[i])
		i+=1
	output_stream.stop()
	

def test_network_playback():
	Url = 'rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov'
	try:
		output_stream = VideoGear(source = Url).start()
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

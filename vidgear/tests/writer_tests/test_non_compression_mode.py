from vidgear.gears import WriteGear
import sys
import numpy as np
import os
from numpy.testing import assert_equal
from vidgear.gears.helper import capPropId
from vidgear.gears.helper import check_output
from six import string_types
import pytest
import cv2
import tempfile

def return_static_ffmpeg():
	path = ''
	if os.name == 'nt':
		path += os.path.join(os.environ['USERPROFILE'],'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe')
	else:
		path += os.path.join(os.environ['HOME'],'Downloads/FFmpeg_static/ffmpeg/ffmpeg')
	return os.path.abspath(path)

def return_testvideo_path():
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)

@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize('conversion', ['COLOR_BGR2GRAY', '', 'COLOR_BGR2YUV', 'COLOR_BGR2BGRA', 'COLOR_BGR2RGB', 'COLOR_BGR2RGBA'])
def test_write(conversion):
	stream = cv2.VideoCapture(return_testvideo_path()) #Open live webcam video stream on first index(i.e. 0) device
	writer = WriteGear(output_filename = 'Output_twc.avi', compression_mode = False) #Define writer
	while True:
		(grabbed, frame) = stream.read()
		# read frames
		# check if frame empty
		if not grabbed:
			#if True break the infinite loop
			break
		if conversion:
			frame = cv2.cvtColor(frame, capPropId(conversion))
		writer.write(frame)
	stream.release()
	writer.close()
	basepath, _ = os.path.split(return_static_ffmpeg()) #extract file base path for debugging aheadget
	ffprobe_path  = os.path.join(basepath,'ffprobe.exe' if os.name == 'nt' else 'ffprobe')
	result = check_output([ffprobe_path, "-v", "error", "-count_frames", "-i", os.path.abspath('Output_twc.avi')])
	if result:
		if not isinstance(result, string_types)
		    result = result.decode()
		print('Result: {}'.format(result))
		for i in ["Error", "Invalid", "error", "invalid"]:
			assert not(i in result)
	os.remove(os.path.abspath('Output_twc.avi'))
	

test_data_class = [
	('', {}, False),
	('Output_twc.avi', {}, True),
	(tempfile.gettempdir(), {}, True),
	('Output_twc.mp4', {"-fourcc":"DIVX"}, True)]
@pytest.mark.parametrize('f_name, output_params, result', test_data_class)
def test_WriteGear_compression(f_name, output_params, result):
	try:
		stream = cv2.VideoCapture(return_testvideo_path()) #Open live webcam video stream on first index(i.e. 0) device
		writer = WriteGear(output_filename = f_name, compression_mode = False , logging = True, **output_params)
		while True:
			(grabbed, frame) = stream.read()
			if not grabbed:
				break
			writer.write(frame)
		stream.release()
		writer.close()
		if f_name and f_name != tempfile.gettempdir():
			os.remove(os.path.abspath(f_name))
	except Exception as e:
		if result:
			pytest.fail(str(e))
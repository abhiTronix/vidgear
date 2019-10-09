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

from vidgear.gears import WriteGear
from vidgear.gears.helper import capPropId
from vidgear.gears.helper import check_output
from six import string_types

import os, platform
import pytest
import cv2
import tempfile



def return_static_ffmpeg():
	"""
	return FFmpeg static path
	"""
	path = ''
	if platform.system() == 'Windows':
		path += os.path.join(tempfile.gettempdir(),'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe')
	elif platform.system() == 'Darwin':
		path += os.path.join(tempfile.gettempdir(),'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg')
	else:
		path += os.path.join(tempfile.gettempdir(),'Downloads/FFmpeg_static/ffmpeg/ffmpeg')
	return os.path.abspath(path)



def return_testvideo_path():
	"""
	return Test Video Data path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
	return os.path.abspath(path)




@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize('conversion', ['COLOR_BGR2GRAY', '', 'COLOR_BGR2YUV', 'COLOR_BGR2BGRA', 'COLOR_BGR2RGB', 'COLOR_BGR2RGBA'])
def test_write(conversion):
	"""
	Testing VidGear Non-Compression(OpenCV) Mode Writer
	"""
	stream = cv2.VideoCapture(return_testvideo_path())
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
	basepath, _ = os.path.split(return_static_ffmpeg())
	ffprobe_path  = os.path.join(basepath,'ffprobe.exe' if os.name == 'nt' else 'ffprobe')
	result = check_output([ffprobe_path, "-v", "error", "-count_frames", "-i", os.path.abspath('Output_twc.avi')])
	if result:
		if not isinstance(result, string_types):
			result = result.decode()
		print('[LOG]: Result: {}'.format(result))
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
	"""
	Testing VidGear Non-Compression(OpenCV) Mode with different parameters
	"""
	try:
		stream = cv2.VideoCapture(return_testvideo_path())
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
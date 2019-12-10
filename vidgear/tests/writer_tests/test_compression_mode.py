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

import pytest
import cv2
import tempfile
import os, platform
import subprocess, re



def return_static_ffmpeg():
	"""
	returns system specific FFmpeg static path
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
	returns Test video path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
	return os.path.abspath(path)



def getFrameRate(path):
	"""
	Returns framerate of video(at path provided) using FFmpeg
	"""
	process = subprocess.Popen([return_static_ffmpeg(), "-i", path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	stdout, _ = process.communicate()
	output =  stdout.decode()
	match_dict = re.search(r"\s(?P<fps>[\d\.]+?)\stbr", output).groupdict()
	return float(match_dict["fps"])



@pytest.mark.xfail(raises=AssertionError)
def test_input_framerate():
	"""
	Testing "-input_framerate" parameter provided by WriteGear(in Compression Mode) 
	"""
	stream = cv2.VideoCapture(return_testvideo_path()) #Open stream
	test_video_framerate = stream.get(cv2.CAP_PROP_FPS)
	output_params = {"-input_framerate":test_video_framerate}
	writer = WriteGear(output_filename = 'Output_tif.mp4', custom_ffmpeg = return_static_ffmpeg(), **output_params) #Define writer 
	while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
		writer.write(frame) 
	stream.release()
	writer.close()
	output_video_framerate = getFrameRate(os.path.abspath('Output_tif.mp4'))
	assert test_video_framerate == output_video_framerate
	os.remove(os.path.abspath('Output_tif.mp4'))



@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize('conversion', ['COLOR_BGR2GRAY', '', 'COLOR_BGR2YUV', 'COLOR_BGR2BGRA', 'COLOR_BGR2RGB', 'COLOR_BGR2RGBA'])
def test_write(conversion):
	"""
	Testing WriteGear Compression-Mode(FFmpeg) Writer capabilties in different colorspace
	"""
	stream = cv2.VideoCapture(return_testvideo_path()) #Open stream
	writer = WriteGear(output_filename = 'Output_tw.mp4',  custom_ffmpeg = return_static_ffmpeg()) #Define writer
	while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
		if conversion:
			frame = cv2.cvtColor(frame, capPropId(conversion))
		if conversion in ['COLOR_BGR2RGB', 'COLOR_BGR2RGBA']:
			writer.write(frame, rgb_mode = True)
		else:
			writer.write(frame)
	stream.release()
	writer.close()
	basepath, _ = os.path.split(return_static_ffmpeg()) 
	ffprobe_path  = os.path.join(basepath,'ffprobe.exe' if os.name == 'nt' else 'ffprobe')
	result = check_output([ffprobe_path, "-v", "error", "-count_frames", "-i", os.path.abspath('Output_tw.mp4')])
	if result:
		if not isinstance(result, string_types):
			result = result.decode()
		print('[LOG]: Result: {}'.format(result))
		for i in ["Error", "Invalid", "error", "invalid"]:
			assert not(i in result)
	os.remove(os.path.abspath('Output_tw.mp4'))



@pytest.mark.xfail(raises=AssertionError)
def test_output_dimensions():
	"""
	Testing "-output_dimensions" special parameter provided by WriteGear(in Compression Mode) 
	"""
	dimensions = (640,480)
	stream = cv2.VideoCapture(return_testvideo_path()) 
	output_params = {"-output_dimensions":dimensions}
	writer = WriteGear(output_filename = 'Output_tod.mp4',  custom_ffmpeg = return_static_ffmpeg(), logging = True, **output_params) #Define writer
	while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
		writer.write(frame)
	stream.release()
	writer.close()
	
	output = cv2.VideoCapture(os.path.abspath('Output_tod.mp4'))
	output_dim = (output.get(cv2.CAP_PROP_FRAME_WIDTH), output.get(cv2.CAP_PROP_FRAME_HEIGHT))
	assert output_dim[0] == 640 and output_dim[1] == 480
	output.release()

	os.remove(os.path.abspath('Output_tod.mp4'))



test_data_class = [
	('','', {}, False),
	('Output1.mp4','', {}, True),
	(tempfile.gettempdir(),'', {}, True),
	('Output2.mp4','', {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}, True),
	('Output3.mp4', return_static_ffmpeg(), {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}, True),
	('Output4.mp4','wrong_test_path', {" -vcodec  ":" libx264", "   -crf": 0, "-preset    ": " fast "}, False)]

@pytest.mark.parametrize('f_name, c_ffmpeg, output_params, result', test_data_class)
def test_WriteGear_compression(f_name, c_ffmpeg, output_params, result):
	"""
	Testing WriteGear Compression-Mode(FFmpeg) with different parameters
	"""
	try:
		stream = cv2.VideoCapture(return_testvideo_path()) #Open stream
		writer = WriteGear(output_filename = f_name, compression_mode = True, **output_params)
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



@pytest.mark.xfail(raises=AssertionError)
def test_WriteGear_customFFmpeg():
	"""
	Testing WriteGear Compression-Mode(FFmpeg) custom FFmpeg Pipeline by seperating audio from video
	"""
	output_audio_filename = 'input_audio.aac'

	#define writer
	writer = WriteGear(output_filename = 'Output.mp4', logging = True) #Define writer 

	#save stream audio as 'input_audio.aac'
	ffmpeg_command_to_save_audio = ['-y', '-i', return_testvideo_path(), '-vn', '-acodec', 'copy', output_audio_filename]
	# `-y` parameter is to overwrite outputfile if exists

	#execute FFmpeg command
	writer.execute_ffmpeg_cmd(ffmpeg_command_to_save_audio)

	#assert audio file is created successfully
	assert os.path.isfile(output_audio_filename) 

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
import pytest
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear
from .fps import FPS



def return_testvideo_path():
	"""
	return Test Video Data path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny.mp4'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)



def return_static_ffmpeg():
	"""
	return FFmpeg static path
	"""
	path = ''
	if os.name == 'nt':
		path += os.path.join(os.environ['USERPROFILE'],'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe')
	else:
		path += os.path.join(os.environ['HOME'],'Downloads/FFmpeg_static/ffmpeg/ffmpeg')
	return os.path.abspath(path)



def Videowriter_non_compression_mode(path):
	"""
	Function to Benchmark VidGearwriter - (Non-Compression Mode: OpenCV)
	"""
	options = {'THREADED_QUEUE_MODE':False}
	stream = VideoGear(source=path, **options).start() 
	writer = WriteGear(output_filename = 'Output_vnc.mp4', compression_mode = False )
	fps_CV = FPS().start()
	while True:
		frame = stream.read()
		if frame is None:
			break
		writer.write(frame)
		fps_CV.update()
	fps_CV.stop()
	stream.stop()
	writer.close()
	print("OpenCV Writer")
	print("[LOG] total elasped time: {:.2f}".format(fps_CV.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps_CV.fps()))
	os.remove(os.path.abspath('Output_vnc.mp4'))



def Videowriter_compression_mode(path):
	"""
	Function to Benchmark VidGearwriter - (Compression Mode: FFmpeg)
	"""
	options = {'THREADED_QUEUE_MODE':False}
	stream = VideoGear(source=path, **options).start()
	writer = WriteGear(output_filename = 'Output_vc.mp4', custom_ffmpeg = return_static_ffmpeg())
	fps_Vid = FPS().start()
	while True:
		frame = stream.read()
		if frame is None:
			break
		writer.write(frame)
		fps_Vid.update()
	fps_Vid.stop()
	stream.stop()
	writer.close()
	print("FFmpeg Writer")
	print("[LOG] total elasped time: {:.2f}".format(fps_Vid.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps_Vid.fps()))
	os.remove(os.path.abspath('Output_vc.mp4'))


@pytest.mark.xfail(raises=RuntimeError)
def test_benchmark_videowriter():
	"""
	Benchmarking VidGearwriter - (Compression Mode: FFmpeg) against (Non-Compression Mode: OpenCV)
	"""
	try:
		Videowriter_non_compression_mode(return_testvideo_path())
		Videowriter_compression_mode(return_testvideo_path())
	except Exception as e:
		raise RuntimeError(e)

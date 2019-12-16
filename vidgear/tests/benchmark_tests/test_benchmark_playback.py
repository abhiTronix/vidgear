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

import os, platform
import pytest
import tempfile
from vidgear.gears import CamGear
from .fps import FPS



def return_testvideo(level=0):
	"""
	returns test H264 videos path with increasing Video quality(resolution & bitrate) with Level-0(Lowest ~HD 2Mbps) and Level-5(Highest ~4k UHD 120mpbs)
	"""
	Levels = ['BigBuckBunny.mp4','20_mbps_hd_hevc_10bit.mkv','50_mbps_hd_h264.mkv','90_mbps_hd_hevc_10bit.mkv','120_mbps_4k_uhd_h264.mkv']
	path = '{}/Downloads/Test_videos/{}'.format(tempfile.gettempdir(), Levels[level])
	return os.path.abspath(path)



def playback(level):
	"""
	tests CamGear API's playback capabilities
	"""
	options = {'THREADED_QUEUE_MODE':False}
	stream = CamGear(source=level, **options).start()
	fps = FPS().start()
	while True:
		frame = stream.read()
		if frame is None:
			break
		fps.update()
	stream.stop()
	fps.stop()
	print("[LOG] total elasped time: {:.2f}".format(fps.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps.fps()))



@pytest.mark.parametrize('level', [return_testvideo(0), return_testvideo(1), return_testvideo(2),return_testvideo(3),return_testvideo(4)])
def test_benchmark(level):
	"""
	Benchmarks low to extreme 4k video playback capabilities of CamGear API
	"""
	if platform.system() != 'Darwin':
		try:
			playback(level)
		except Exception as e:
			print(e)
	else:
		print("Skipping this test for macOS!")

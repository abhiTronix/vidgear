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


import pytest
from vidgear.gears import VideoGear



def test_PiGear_import():
	"""
	Testing VideoGear Import - made to fail when PiGear class is imported
	"""
	with pytest.raises(ImportError):
		stream = VideoGear(enablePiCamera = True, logging = True).start()
		stream.stop()



def test_CamGear_import():
	"""
	Testing VideoGear Import - Passed if CamGear Class is Imported sucessfully
	"""
	try:
		Url = 'rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov'
		options = {'THREADED_QUEUE_MODE':False}
		output_stream = VideoGear(source = Url, **options).start()
		output_stream.stop()
	except Exception as e:
		pytest.fail(str(e))

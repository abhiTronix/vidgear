"""
Copyright (c) 2019 Abhishek Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""



import os
import pytest
from vidgear.gears import CamGear
from .fps import FPS



def return_testvideo(level=0):
	"""
	return Test Video Data path with different Video quality/resolution/bitrate for different levels(Level-0(Lowest ~HD 2Mbps) and Level-5(Highest ~4k UHD 120mpbs))
	"""
	Levels = ['BigBuckBunny.mp4','20_mbps_hd_hevc_10bit.mkv','50_mbps_hd_h264.mkv','90_mbps_hd_hevc_10bit.mkv','120_mbps_4k_uhd_h264.mkv']
	path = '{}/Downloads/Test_videos/{}'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'], Levels[level])
	return os.path.abspath(path)



def playback(level):
	"""
	Function to test VidGear playback capabilities
	"""
	stream = CamGear(source=level).start()
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
	Benchmarking low to extreme 4k video playback capabilities of VidGear
	"""
	try:
		playback(level)
	except Exception as e:
		print(e)

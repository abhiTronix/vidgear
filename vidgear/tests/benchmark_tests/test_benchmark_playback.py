import os
import pytest
from vidgear.gears import CamGear
from .fps import FPS

def return_testvideo(level=0):
	Levels = ['BigBuckBunny.mp4','20_mbps_hd_hevc_10bit.mkv','50_mbps_hd_h264.mkv','90_mbps_hd_hevc_10bit.mkv','120_mbps_4k_uhd_h264.mkv']
	path = '{}/download/Test_videos/{}'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'], Levels[level])
	return os.path.abspath(path)

def playback(level):
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
	try:
		playback(level)
	except Exception as e:
		print(e)

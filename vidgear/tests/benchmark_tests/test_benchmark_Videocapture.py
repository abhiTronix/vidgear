import os
import cv2
import pytest
from vidgear.gears import CamGear
from .fps import FPS

def return_testvideo_path():
	path = '{}/Downloads/Test_videos/BigBuckBunny.mp4'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)

def Videocapture_withCV(path):
	stream = cv2.VideoCapture(path)
	fps_CV = FPS().start()
	while True:
		(grabbed, frame) = stream.read()
		if not grabbed:
			break
		fps_CV.update()
	fps_CV.stop()
	stream.release()
	print("OpenCV")
	print("[LOG] total elasped time: {:.2f}".format(fps_CV.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps_CV.fps()))

def Videocapture_withVidGear(path):
	stream = CamGear(source=path).start()
	fps_Vid = FPS().start()
	while True:
		frame = stream.read()
		if frame is None:
			break
		fps_Vid.update()
	fps_Vid.stop()
	stream.stop()
	print("VidGear")
	print("[LOG] total elasped time: {:.2f}".format(fps_Vid.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps_Vid.fps()))


def test_benchmark_videocapture():
	try:
		Videocapture_withCV(return_testvideo_path())
		Videocapture_withVidGear(return_testvideo_path())
	except Exception as e:
		print(e)

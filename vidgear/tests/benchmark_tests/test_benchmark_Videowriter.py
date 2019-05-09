import os
import pytest
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear
from .fps import FPS

def return_testvideo_path():
	path = '{}/Downloads/Test_videos/BigBuckBunny.mp4'.format(os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'])
	return os.path.abspath(path)

def return_static_ffmpeg():
	path = ''
	if os.name == 'nt':
		path += os.path.join(os.environ['USERPROFILE'],'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe')
	else:
		path += os.path.join(os.environ['HOME'],'Downloads/FFmpeg_static/ffmpeg/ffmpeg')
	return os.path.abspath(path)

def Videowriter_non_compression_mode(path):
	stream = VideoGear(source=path).start() 
	writer = WriteGear(output_filename = 'Output.mp4', compression_mode = False ) #Define writer with output filename 'Output.mp4'
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
	os.remove(os.path.abspath('Output.mp4'))

def Videowriter_compression_mode(path):
	stream = VideoGear(source=path).start()
	writer = WriteGear(output_filename = 'Output.mp4', custom_ffmpeg = return_static_ffmpeg()) #Define writer with output filename 'Output.mp4'
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
	print("VidGear Writer")
	print("[LOG] total elasped time: {:.2f}".format(fps_Vid.total_time_elapsed()))
	print("[LOG] approx. FPS: {:.2f}".format(fps_Vid.fps()))
	os.remove(os.path.abspath('Output.mp4'))


def test_benchmark_videowriter():
	try:
		Videowriter_non_compression_mode(return_testvideo_path())
		Videowriter_compression_mode(return_testvideo_path())
	except Exception as e:
		print(e)

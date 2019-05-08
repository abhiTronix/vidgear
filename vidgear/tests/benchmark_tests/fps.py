#modified fps module from `imutils`
import time

class FPS:
	def __init__(self):
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		self._start = time.time()
		return self

	def stop(self):
		self._end = time.time()

	def frame_update(self):
		self._numFrames += 1

	def total_time_elapsed(self):
		return (self._end - self._start)%60

	def fps(self):
		return self._numFrames / self.total_time_elapsed()
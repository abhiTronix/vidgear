#This class is a modified fps module from `imutils` python library.

import time

class FPS:
	"""
	Class to calculate FPS based on time.time python module
	"""
	def __init__(self):
		#initiating FPS class and its variable
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		#start timer
		self._start = time.time()
		return self

	def stop(self):
		#stop timer
		self._end = time.time()

	def update(self):
		#calculate frames
		self._numFrames += 1

	def total_time_elapsed(self):
		#total time elaspsed = start time - end time
		return (self._end - self._start)%60

	def fps(self):
		#return FPS
		return self._numFrames / self.total_time_elapsed()
# This class is a modified fps module from `imutils` python library.

import time


class FPS:
    """
	Class to calculate FPS based on time.time python module
	"""

    def __init__(self):
        # initiating FPS class and its variable
        self._start = 0
        self._end = 0

    def start(self):
        # start timer
        self._start = time.time()
        return self

    def stop(self):
        # stop timer
        self._end = time.time()

    def total_time_elapsed(self):
        # return total time elaspsed = start time - end time(in sec)
        if self._end <= self._start:
            self._end = time.time()
        return (self._end - self._start)

    def fps(self):
        # return FPS
        return 1.0 / self.total_time_elapsed()
